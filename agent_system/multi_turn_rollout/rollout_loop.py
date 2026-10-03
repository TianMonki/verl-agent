# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from recipe.sgpo.rewind import plan_rewind, summarize_trajectories

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.

        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        # Filled in by vanilla_multi_turn_loop when CBPO branching is on; read by the trainer for
        # logging. None means the last rollout was plain i.i.d. sampling.
        self.branch_stats: Dict[str, float] | None = None
        # Filled in by multi_turn_loop when CRB rewind branching is on; same contract.
        self.rewind_stats: Dict[str, float] | None = None
        # Wall-clock split of the training rollout, accumulated over every env step of every phase
        # and read by the trainer. timing_s/gen is 60-65% of a WebShop step and used to be a single
        # opaque number covering three very different things: driver-side prompt tokenisation, the
        # vLLM generate call, and envs.step (128 ray workers each holding a pyserini JVM). None means
        # the last rollout was validation, which is not accounted.
        self.rollout_timing: Dict[str, float] | None = None

    def _timing_add(self, key: str, seconds: float) -> None:
        if self.rollout_timing is not None:
            self.rollout_timing[key] = self.rollout_timing.get(key, 0.0) + seconds

    def _make_branch_scheduler(self, batch_size: int, is_train: bool):
        """A BranchScheduler if CBPO prefix sharing is enabled for this rollout, else None."""
        cfg = self.config.env.rollout.get("branch", None)
        if not is_train or cfg is None or not cfg.get("enable", False):
            return None
        group_n = int(self.config.env.rollout.n)
        if group_n < 2 or batch_size % group_n != 0:
            raise ValueError(
                f"env.rollout.branch.enable requires env.rollout.n>=2 dividing the rollout batch, "
                f"got n={group_n} batch={batch_size}."
            )
        if self.config.algorithm.filter_groups.enable:
            # DAPO's dynamic sampling throws away exactly the zero-variance groups that prefix
            # sharing exists to make non-degenerate, and it would re-enter vanilla_multi_turn_loop
            # with a fresh scheduler per attempt, so the two are mutually exclusive.
            raise ValueError("env.rollout.branch.enable is incompatible with algorithm.filter_groups.enable")
        raise NotImplementedError(
            "env.rollout.branch (CBPO prefix-sharing) was removed in the SGPO-only release. "
            "SGPO uses counterfactual rewind branching (CRB) via env.rollout.rewind instead."
        )

    def _rewind_cfg(self, is_train: bool):
        """The CRB config if rewind branching applies to this rollout, else None."""
        cfg = self.config.env.rollout.get("rewind", None)
        if not is_train or cfg is None or not cfg.get("enable", False):
            return None
        if self.config.algorithm.filter_groups.enable:
            # Dynamic sampling resamples whole batches and hands multi_turn_loop a trajectory list
            # whose length is no longer batch_size, so the row->worker mapping CRB replays against
            # does not exist. It also discards exactly the degenerate groups CRB exists to repair.
            raise ValueError("env.rollout.rewind.enable is incompatible with algorithm.filter_groups.enable")
        return cfg

    def _generate_rows(self, batch_input: DataProto, rows: np.ndarray, meta_info, actor_rollout_wg):
        """Run generation for a subset of rows, returning an output of exactly ``len(rows)`` rows."""
        gen_input = batch_input.select_idxs(rows)
        gen_input.meta_info = meta_info
        padded, pad_size = pad_dataproto_to_divisor(gen_input, actor_rollout_wg.world_size)
        return unpad_dataproto(actor_rollout_wg.generate_sequences(padded), pad_size=pad_size)

    @staticmethod
    def _mean_surprisal(output: DataProto) -> np.ndarray:
        """Mean per-token ``-log p`` of each generated action under the rollout policy.

        ``rollout_log_probs`` is padded with -1 past the end of the response
        (``vllm_rollout_spmd.py:344``), so the response half of ``attention_mask`` is used as the
        mask rather than trusting the padding value.
        """
        lp = output.batch["rollout_log_probs"].float()
        resp_len = output.batch["responses"].shape[-1]
        mask = output.batch["attention_mask"][:, -resp_len:].to(lp.dtype)
        n = mask.sum(-1).clamp(min=1.0)
        return (-(lp * mask).sum(-1) / n).cpu().numpy()

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            rewind=None,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
            rewind (RewindPlan): CRB plan. When given, the envs are reset to the plan's goals rather
                than to a fresh draw, the plan's rows replay a recorded action prefix before they
                start generating, and every other row is inert. See recipe/sgpo/rewind.py.

        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        if rewind is None:
            obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        else:
            if not hasattr(envs, 'reset_to'):
                raise NotImplementedError(
                    "env.rollout.rewind.enable needs a replayable env manager exposing reset_to(); "
                    f"{type(envs).__name__} does not."
                )
            obs, infos = envs.reset_to(rewind.goal_idxs)

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if rewind is not None:
            # CRB: the continuations of one replayed prefix form their own group, with a fresh uid
            # per prefix. The source group's uid is deliberately *not* reused -- see the "credit
            # assignment is settled by construction" note in recipe/sgpo/rewind.py.
            uid_batch = rewind.uid.copy()
        elif self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        if rewind is not None:
            # Rows the plan did not use are inert for the whole phase: they are marked done up front
            # so they never generate and gather_rollout_data drops them.
            is_done |= ~rewind.active
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        branch = None if rewind is not None else self._make_branch_scheduler(batch_size, is_train)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            if rewind is not None:
                # A row still replaying its recorded prefix has to step the env -- that is what puts
                # it into the forked state -- but its action is scripted, so it neither generates nor
                # enters the training batch.
                active_masks = active_masks & ~rewind.replaying(_step)

            prep_t0 = time.perf_counter()
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            self._timing_add("timing_s/gen_prep", time.perf_counter() - prep_t0)
            gen_t0 = time.perf_counter()

            # Rows whose episode already finished are dropped in gather_rollout_data (which keeps
            # only active_masks==True steps) and are skipped by the success evaluator, so their
            # generations are pure waste -- on WebShop roughly half the rollout budget. Generate for
            # the active rows only, then gather the result back to full width so everything
            # downstream (envs.step, to_list_of_dict, index alignment with total_infos) is unchanged.
            # Inactive rows receive a copy of the first active row's output; it is never trained on.
            if branch is not None:
                # CBPO: during the shared prefix only the group leader generates, and its response
                # tokens are broadcast to the followers, which therefore execute the leader's action
                # and stay in a bit-identical env state. Two passes, because whether to fork can
                # depend on how surprising the leader's action turned out to be.
                pass1 = branch.plan(_step, active_masks)
                out1 = self._generate_rows(batch_input, pass1, gen_batch.meta_info, actor_rollout_wg)
                surprisal = np.full(batch_size, np.nan, dtype=np.float64)
                surprisal[pass1] = self._mean_surprisal(out1)
                pass2 = branch.decide(_step, surprisal, active_masks)
                if len(pass2):
                    out2 = self._generate_rows(batch_input, pass2, gen_batch.meta_info, actor_rollout_wg)
                    combined = DataProto.concat([out1, out2])
                    generated = np.concatenate([pass1, pass2])
                else:
                    combined, generated = out1, pass1
                batch_output = combined.select_idxs(branch.gather(generated, active_masks))
            else:
                active_idx = np.nonzero(active_masks)[0]
                if len(active_idx) == 0:
                    # Every live row is still replaying a scripted prefix (CRB's opening steps).
                    # Generate one throwaway row instead of handing the worker group an empty batch;
                    # its output is overwritten by the script below and is never trained on.
                    active_idx = np.zeros(1, dtype=np.int64)
                if len(active_idx) < batch_size:
                    gen_input = batch_input.select_idxs(active_idx)
                    gen_input.meta_info = gen_batch.meta_info
                else:
                    gen_input = batch_input

                # pad to be divisible by dp_size
                batch_input_padded, pad_size = pad_dataproto_to_divisor(gen_input, actor_rollout_wg.world_size)
                batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
                # # unpad
                batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

                if len(active_idx) < batch_size:
                    scatter = np.zeros(batch_size, dtype=np.int64)
                    scatter[active_idx] = np.arange(len(active_idx), dtype=np.int64)
                    batch_output = batch_output.select_idxs(scatter)

            self._timing_add("timing_s/gen_llm", time.perf_counter() - gen_t0)
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            if rewind is not None:
                # Replace the replaying rows' freshly generated action with the recorded one. Only
                # the string handed to envs.step matters here: those rows are excluded from
                # active_masks, so the generated tokens they carry are dropped before training.
                text_actions = rewind.script_actions(_step, text_actions)

            env_t0 = time.perf_counter()
            next_obs, rewards, dones, infos = envs.step(text_actions)
            self._timing_add("timing_s/gen_env", time.perf_counter() - env_t0)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
            if 'task_score' in infos[0]:
                batch.non_tensor_batch['task_score'] = np.array([info.get('task_score', 0.0) for info in infos], dtype=np.float32)
            if 'req_scores' in infos[0]:
                # Per-requirement decomposition of the WebShop score, present only on the row where
                # the episode terminated with a purchase (None otherwise). Object dtype because the
                # payload is a dict; dropped by the actor's select_keys, consumed only by RGPO.
                batch.non_tensor_batch['req_scores'] = np.array(
                    [info.get('req_scores') for info in infos], dtype=object
                )
            if 'caa_adv' in infos[0]:
                # Counterfactual advantage of the product the agent clicked over the results page it
                # was shown, from the env's own reward function evaluated on the actions not taken.
                # 0.0 on every step that was not a click from a multi-candidate page.
                batch.non_tensor_batch['caa_adv'] = np.array(
                    [info.get('caa_adv', 0.0) for info in infos], dtype=np.float32
                )
            if 'cvp_pass' in infos[0]:
                # Observable constraint verdict on a completed purchase: 1.0 the bought product offers
                # every option value the instruction names, 0.0 it does not, -1.0 nothing to grade.
                # No reward function and no target identity are consulted -- only the instruction and
                # the option buttons the agent was shown.
                batch.non_tensor_batch['cvp_pass'] = np.array(
                    [info.get('cvp_pass', -1.0) for info in infos], dtype=np.float32
                )
            if 'goal_idx' in infos[0]:
                # Goal identity, needed by GDPO's per-goal difficulty tracker. The agent-visible
                # observation has the instruction stripped, so this is the only goal discriminator.
                batch.non_tensor_batch['goal_idx'] = np.array(
                    [info.get('goal_idx', -1) for info in infos], dtype=np.int64
                )
            if 'goal_key' in infos[0]:
                # '|'-separated hierarchical difficulty key, most specific level first.
                batch.non_tensor_batch['goal_key'] = np.array(
                    [info.get('goal_key') for info in infos], dtype=object
                )

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            batch.non_tensor_batch['text_actions'] = np.array(text_actions, dtype=object)
            batch.non_tensor_batch['step_id'] = np.full(batch_size, _step, dtype=np.int64)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            if rewind is not None:
                # A scripted prefix step must never terminate the episode: the prefix stops at least
                # one action short of where the source trajectory ended. A non-zero count here means
                # the replay diverged, i.e. the purity assumption CRB rests on is violated.
                rewind.note_dones(_step, dones)
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        if branch is not None:
            self.branch_stats = branch.stats(episode_rewards)

        if rewind is not None:
            # The default evaluator asserts one graded row per trajectory, which the inert rows of a
            # CRB phase cannot supply. The caller reuses the main rollout's success dict instead, so
            # the logged episode metrics keep meaning "the on-policy rollout", and the branches are
            # reported separately under rewind/*.
            success: Dict[str, np.ndarray] = {}
        else:
            success: Dict[str, np.ndarray] = envs.success_evaluator(
                        total_infos=total_infos,
                        total_batch_list=total_batch_list,
                        episode_rewards=episode_rewards,
                        episode_lengths=episode_lengths,
                        )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def rewind_phase(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            cfg,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            success: Dict[str, np.ndarray],
            ) -> DataProto | None:
        """CRB: replay the degenerate groups' prefixes and re-sample their last `depth` steps.

        Returns the extra trajectories as a DataProto to be concatenated onto the on-policy rollout,
        or None when there was nothing to branch. `success` is the *main* rollout's success dict and
        is passed straight through to gather_rollout_data, which only ever takes its mean -- so the
        branch rows carry the same episode-level logging constants as the rows they were forked from
        and the logged success_rate keeps meaning "the on-policy rollout".
        """
        group_n = int(self.config.env.rollout.n)
        goal_idx, won, dense, lengths, actions = summarize_trajectories(total_batch_list, episode_rewards)

        stats: Dict[str, float] = {}
        out: List[DataProto] = []
        # Rows eligible as prefix sources. Round 1 may fork any trajectory of a degenerate group;
        # later rounds only re-fork the branch groups that came out degenerate anyway, one step
        # deeper -- which is the adaptive-depth part, and why the source pool shrinks per round.
        restrict = None
        max_rounds = max(int(cfg.get("max_rounds", 1)), 1)
        for rnd in range(max_rounds):
            plan = plan_rewind(
                group_n=group_n, goal_idx=goal_idx, won=won, dense=dense, lengths=lengths,
                actions=actions, n_prefix=int(cfg.get("n_prefix", 4)),
                n_branch=int(cfg.get("n_branch", 2)),
                depth=int(cfg.get("depth", 1)) + rnd,
                only_all_lose=bool(cfg.get("only_all_lose", True)),
                restrict_rows=restrict,
            )
            if rnd == 0:
                stats.update(plan.plan_stats())
            if not plan.any():
                break

            bl, er, el, _, tu, tc = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs,
                is_train=True, rewind=plan,
            )
            round_stats = plan.outcome_stats(er)
            if rnd == 0:
                stats.update(round_stats)
            else:
                stats.update({f"{k}_r{rnd}": v for k, v in round_stats.items()})

            # A branch group whose members all ended on the same episode reward carries no contrast,
            # which is the only thing it was created for: its group-centered episode advantage is
            # exactly 0, and an all-lose branch group has reward 0 everywhere so its step residual is
            # 0 too. Measured on one WebShop run, ~92% of branch groups came out flat, i.e. most of
            # the extra rows were only paying padding. Drop them.
            flat = plan.degenerate_branch_uids(er)
            drop_flat = bool(cfg.get("drop_degenerate", True))
            keep = [i for i in np.nonzero(plan.active)[0]
                    if (not drop_flat or plan.uid[int(i)] not in flat)
                    and any(row['active_masks'] for row in bl[i])]
            if keep:
                out.append(self.gather_rollout_data(
                    total_batch_list=[bl[i] for i in keep],
                    episode_rewards=er[keep],
                    episode_lengths=el[keep],
                    success=success,
                    traj_uid=tu[keep],
                    tool_callings=tc[keep],
                ))

            if rnd + 1 < max_rounds:
                # Re-fork only the source trajectories whose branch group stayed degenerate.
                restrict = {int(plan.src_row[i]) for i in np.nonzero(plan.active)[0]
                            if plan.uid[int(i)] in flat}
                if not restrict:
                    break

        stats["rewind/n_extra_rows"] = float(sum(len(o.batch) for o in out))
        self.rewind_stats = stats
        if not out:
            return None
        return out[0] if len(out) == 1 else DataProto.concat(out)

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        # Accumulate over every phase of this rollout (dynamic sampling attempts and the CRB rewind
        # phase included), so the three keys sum to what the trainer reports as timing_s/gen. Off for
        # validation, whose cost is already reported separately as timing_s/testing.
        self.rollout_timing = {} if is_train else None

        rewind_cfg = self._rewind_cfg(is_train)

        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )

        if rewind_cfg is not None:
            # CRB runs after the on-policy rollout because it needs the finished trajectories: which
            # groups came out degenerate, and what actions to replay. It resets the envs, so nothing
            # above may depend on their post-rollout state.
            extra = self.rewind_phase(
                gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs, cfg=rewind_cfg,
                total_batch_list=total_batch_list, episode_rewards=total_episode_rewards,
                success=total_success,
            )
            if extra is not None:
                gen_batch_output = DataProto.concat([gen_batch_output, extra])

        return gen_batch_output
