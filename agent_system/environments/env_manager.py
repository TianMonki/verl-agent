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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


def maybe_append_assess_suffix(config, obs: str) -> str:
    """Append the VVPO <assess> format suffix, if enabled.

    Applied *after* template formatting so that agent_system/environments/prompts/*.py stay
    byte-identical and every other estimator's baseline remains comparable. Env-agnostic, so
    the same hook works for ALFWorld.
    """
    algo = getattr(config, "algorithm", None)
    vvpo = None if algo is None else algo.get("vvpo", None)
    if not vvpo or not bool(vvpo.get("enable_prompt", False)):
        return obs
    raise NotImplementedError(
        "algorithm.vvpo.enable_prompt (VVPO assess suffix) was removed in the SGPO-only release."
    )


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.current_available_action_sets = None
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        # Local, like every other env_package import here: alfworld/__init__.py pulls in ray and
        # the textworld stack, which a WebShop run must not have to have installed.
        from agent_system.environments.env_package.alfworld.state_key import ProgressTracker, sound_key
        from agent_system.environments.env_package.alfworld.cvp import CvpTracker
        self._sound_key = sound_key

        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # SGPO's sound state key sigma(s): admissible_commands already covers location/inventory/
        # open-closed state, progress_trackers recovers the heat/cool/clean/turn-on/slice bit that
        # admissible_commands does not change on. Neither reads the gamefile.
        self.progress_trackers = [ProgressTracker() for _ in text_obs]
        admissible = self.envs.get_admissible_commands
        anchor = [str(self._sound_key(admissible[i], self.progress_trackers[i].multiset()))
                  for i in range(len(text_obs))]
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        # CVPO's constraint verdict, one tracker per env. Built from the goal sentence extract_task
        # just pulled out of the observation -- the same string that fills the prompt's
        # {task_description} slot -- so nothing the agent cannot see enters. See cvp.py.
        self.cvp_trackers = [CvpTracker(task) for task in self.tasks]

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': anchor}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        admissible = self.envs.get_admissible_commands
        anchor = []
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            progress = self.progress_trackers[i].update(text_obs[i])
            anchor.append(str(self._sound_key(admissible[i], progress)))
            # Running CVPO verdict: 1.0 the placement satisfies every conjunct of the goal sentence
            # the observation stream can confirm, 0.0 it does not, -1.0 nothing to grade yet. Written
            # on every row so rollout_loop.py's `'cvp_pass' in infos[0]` gate fires; core_rgpo.py
            # keeps the last graded row per trajectory, i.e. the final verdict.
            info['cvp_pass'] = self.cvp_trackers[i].update(text_obs[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': anchor}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.current_available_action_sets = None
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        return self._post_reset(obs, infos)

    def reset_to(self, goal_idxs) -> Dict[str, Any]:
        """Reset to caller-specified goals instead of a fresh random draw.

        Used by CRB (``recipe/sgpo/rewind.py``) to put the envs back on the goals the finished
        rollout used, so recorded action prefixes replay into bit-identical states. Everything after
        the reset call is identical to :meth:`reset`, which matters: ``self.tasks`` (the instruction
        used to strip the goal out of the observation) and ``self.memory`` (the history window in the
        prompt) both have to be rebuilt or the replayed prompts would not match the originals.
        """
        obs, infos = self.envs.reset_to(goal_idxs)
        return self._post_reset(obs, infos)

    def _post_reset(self, obs, infos) -> Dict[str, Any]:
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True),
                        'image': None,
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        if self.config.env.webshop.get("strict_admissible_validation", False):
            valids = self._validate_against_available_actions(actions, valids)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        available_action_sets = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):

            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            available_action_sets.append({action.lower() for action in available_actions})
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)
            current_observation = self._maybe_clip_current_observation(text_obs[i])
            task_state = self._build_task_state(infos[i])

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=current_observation,
                    task_state=task_state,
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=current_observation,
                    task_state=task_state,
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=current_observation,
                        task_state=task_state,
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(maybe_append_assess_suffix(self.config, obs))

        self.current_available_action_sets = available_action_sets
        return postprocess_text_obs

    def _validate_against_available_actions(self, actions: List[str], valids: List[int]) -> List[int]:
        if self.current_available_action_sets is None:
            return valids
        checked = list(valids)
        for i, action in enumerate(actions):
            if i >= len(self.current_available_action_sets) or not checked[i]:
                continue
            action_text = str(action).strip().lower()
            available = self.current_available_action_sets[i]
            has_search = any(item.startswith("search[") for item in available)
            if action_text.startswith("search[") and has_search:
                continue
            if action_text not in available:
                checked[i] = 0
        return checked

    def _maybe_clip_current_observation(self, text: str) -> str:
        webshop_cfg = self.config.env.get("webshop", {})
        max_chars = int(webshop_cfg.get("observation_max_chars", 0) or 0)
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        tail_chars = int(webshop_cfg.get("observation_tail_chars", max_chars // 4) or 0)
        tail_chars = max(0, min(tail_chars, max_chars // 2))
        head_chars = max_chars - tail_chars
        omitted = len(text) - max_chars
        marker = f"\n[... clipped {omitted} chars from long WebShop observation ...]\n"
        if tail_chars == 0:
            return text[:head_chars] + marker
        return text[:head_chars] + marker + text[-tail_chars:]

    def _build_task_state(self, info) -> str:
        """A recap of what the agent itself has already read and clicked this episode.

        It goes into its own template slot rather than into the observation, because ``anchor`` *is*
        ``format_obs(obs)`` (:415/:435) and it is GiGPO's step-grouping key -- per-trajectory text in
        the observation would make almost every anchor unique and silently zero the step-level
        advantage. Returns ``""`` when disabled, which restores the prompt byte-for-byte.

        With ``task_state_evidence_chars > 0`` it also replays the sub-pages the agent read on each
        product. That is what makes comparing candidates possible at all: one inspect cycle costs four
        steps and ``history_length`` is 2, so without the replay the first candidate's description has
        left the context by the time the second is opened. Measured on the live env, reading
        description + features lifts the fraction of a goal's attributes that are literally visible
        from 0.1204 to 0.5554, and a decision made without them caps out at success rate 0.7100.
        """
        webshop_cfg = self.config.env.get("webshop", {})
        if not webshop_cfg.get("task_state", False):
            return ""
        max_viewed = int(webshop_cfg.get("task_state_max_viewed", 6) or 6)
        title_chars = int(webshop_cfg.get("task_state_title_chars", 60) or 60)
        evidence_chars = int(webshop_cfg.get("task_state_evidence_chars", 0) or 0)

        lines = []
        viewed = info.get('viewed_products') or []
        if viewed:
            lines.append(f"Products you have already opened and read ({len(viewed)}):")
            for item in viewed[:max_viewed]:
                title = str(item.get('title', ''))[:title_chars]
                lines.append(f"  {item.get('asin', '')}  {title}  {item.get('price', '')}")
                for page, text in (item.get('evidence') or {}).items():
                    lines.append(f"      {page}: {str(text)[:evidence_chars]}")
            if len(viewed) > max_viewed:
                lines.append(f"  ... and {len(viewed) - max_viewed} more")
        else:
            lines.append("You have not opened any product page yet.")

        options = info.get('selected_options') or {}
        if options:
            rendered = "; ".join(f"{k} = {v}" for k, v in options.items())
            lines.append(f"Options you have selected on the current product: {rendered}")
        else:
            lines.append("You have not selected any option yet.")
        lines.append("Selected options are cleared whenever you run a new search.")
        if evidence_chars > 0:
            # The route exists in the admissible actions of every product page, but 'description' and
            # 'features' look like navigation rather than like the only place the deciding evidence
            # lives, so it is stated once. 'Description'/'Features'/'Reviews' return with '< prev'.
            lines.append(
                "Most of what the instruction asks for is NOT in a product's title: click "
                "'Description' and 'Features' on a product page to read it, then '< prev' to go back. "
                "Read a product before buying it, and compare it against the ones above.")
        if webshop_cfg.get("option_hint", False):
            # Names the route CVPO grades, and nothing else: no verdict is injected, the agent has to
            # read the buttons itself. That is deliberate -- the reading is the behaviour being learned,
            # and it is a behaviour the base model demonstrably has (it already selects 1.70 of the 2.09
            # option values a goal names on the product it buys; it just does so *after* committing to
            # the product instead of using them to choose it). Nothing here is privileged: the option
            # values are printed verbatim in the instruction and the buttons are printed verbatim by
            # click[asin]. Measured payoff of following it: buying rank 1 scores SR 0.5100 and passes
            # this check 0.6500 of the time; buying the first page-1 candidate that passes scores 0.7300.
            lines.append(
                "The instruction names exact option values (a size, a colour, a flavour). A product "
                "page lists every value it offers as buttons before its title. Open a candidate and "
                "check that list: if it does not offer the values the instruction asks for, go 'Back "
                "to Search' and open the next candidate instead of buying this one.")
        return "\n".join(lines) + "\n"

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                success['n_viewed (not success_rate)'].append(float(info.get('n_viewed', 0)))
                success['blocked_buys (not success_rate)'].append(float(info.get('blocked_buys', 0)))
                success['n_options_selected (not success_rate)'].append(
                    float(len(info.get('selected_options') or {})))
                # Retrieval vs. selection failure split: an episode can fail because the target
                # product was never returned by any search the agent ran, or because it was
                # returned and the agent bought something else. No other metric separates these.
                success['target_seen (not success_rate)'].append(
                    float(bool(info.get('target_seen', False))))
                success['target_opened (not success_rate)'].append(
                    float(bool(info.get('target_opened', False))))
                success['n_seen (not success_rate)'].append(float(info.get('n_seen', 0)))
                # Sub-pages (description/features/reviews) actually opened this episode. This is the
                # earliest observable sign of whether deliberation is happening at all: a goal's
                # attributes are 0.1204 visible on a product page as opened and 0.5554 after reading
                # description+features, so an episode that reads nothing cannot exceed SR 0.7100.
                success['n_pages_read (not success_rate)'].append(
                    float(info.get('n_pages_read', 0)))
                # Mean dense score forgone on the results pages this episode clicked from. Measured
                # offline at 0.196 for a policy that always takes rank 1; driving it toward 0 is the
                # whole point of the counterfactual term.
                success['caa_regret (not success_rate)'].append(
                    float(info.get('caa_mean_regret', 0.0)))
                success['caa_clicks (not success_rate)'].append(float(info.get('caa_clicks', 0)))
                # CVPO's own objective, and the earliest sign of whether it is moving: does the product
                # this episode bought offer every option value the instruction names? A policy that
                # buys rank 1 sits at 0.6500 (measured, 200 val goals) and the SR ceiling of that
                # behaviour is 0.7100; buying the first passing page-1 candidate is 0.7300 SR. So this
                # must rise well above 0.65 or the term is not changing behaviour, whatever SR does.
                success['bought_options_ok (not success_rate)'].append(
                    float(info.get('bought_options_ok', 0.0)))
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        min_products_before_buy = int(config.env.webshop.get('min_products_before_buy', 0) or 0)
        caa_std_floor = float(config.env.webshop.get('caa_std_floor', 0.1) or 0.1)
        caa_scope = str(config.env.webshop.get('caa_scope', 'click') or 'click')
        evidence_chars = int(config.env.webshop.get('evidence_chars', 0) or 0)
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker, min_products_before_buy=min_products_before_buy, caa_std_floor=caa_std_floor, caa_scope=caa_scope, evidence_chars=evidence_chars)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker, min_products_before_buy=min_products_before_buy, caa_std_floor=caa_std_floor, caa_scope=caa_scope, evidence_chars=evidence_chars)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
