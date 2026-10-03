# Copyright 2026
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
"""CRB -- Counterfactual Rewind Branching.

Every group-relative estimator (GRPO, GiGPO, SGPO) is a centered deviation inside a group, so on a
group whose members all score the same it returns exactly zero for every row. On WebShop that is the
common case. CRB manufactures the missing within-group outcome variance after the rollout, without
touching the reward function, the prompt, or any privileged information: a WebAgentTextEnv's state is
a pure function of (seed, goal index, action sequence), so a finished trajectory's prefix can be
replayed at no generation cost and only the last few decisions re-sampled.

The mechanism, per degenerate group:

1. Pick the `n_prefix` most promising failed trajectories (highest dense task_score, shorter first).
   Dense score only chooses where to spend compute, never as a reward.
2. Rewind each `depth` steps: replay actions 0 .. L-depth-1 verbatim (env steps, zero tokens).
3. From that fork point, sample `n_branch` independent continuations with the current policy.

The `n_branch` continuations of one prefix form their own group with a fresh uid: they share a
bit-identical prefix, so the only difference is the re-sampled suffix. Replayed prefix rows are not
trained on (active_masks=False), so no token is duplicated into the loss. The original group is left
untouched. Rounds (max_rounds > 1) re-fork one step deeper on branch groups that came out degenerate
anyway; the default is a single round.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

import numpy as np


class RewindPlan:
    """A row-slot assignment for one CRB phase. Pure numpy/str, no torch and no env.

    One instance per rewind round. ``batch_size`` rows keep their original geometry, so row ``i``
    always replays on the worker it belongs to and therefore under the seed its group was created
    with -- the only condition CBPO's determinism check leaves in place ("may only ever branch
    within a group").
    """

    def __init__(self, batch_size, group_n, goal_idxs, active, uid, src_row, prefix_actions,
                 depth, n_degenerate_groups, n_branched_groups):
        self.batch_size = int(batch_size)
        self.group_n = int(group_n)
        self.n_groups = self.batch_size // self.group_n
        self.goal_idxs = np.asarray(goal_idxs, dtype=np.int64)
        self.active = np.asarray(active, dtype=bool)
        self.uid = np.asarray(uid, dtype=object)
        self.src_row = np.asarray(src_row, dtype=np.int64)
        self.prefix_actions = [list(p) for p in prefix_actions]
        self.prefix_len = np.array([len(p) for p in self.prefix_actions], dtype=np.int64)
        self.depth = int(depth)
        self.n_degenerate_groups = int(n_degenerate_groups)
        self.n_branched_groups = int(n_branched_groups)
        self.early_done = 0

    # -- per-step protocol --------------------------------------------------------------------
    def any(self) -> bool:
        return bool(self.active.any())

    def replaying(self, step: int) -> np.ndarray:
        """Rows whose action at ``step`` is scripted from the recorded prefix.

        These rows must still step the env (that is what puts them in the forked state) but they
        neither generate nor enter the training batch.
        """
        return self.active & (int(step) < self.prefix_len)

    def script_actions(self, step: int, text_actions):
        """Overwrite the replaying rows' actions with the recorded ones.

        The recorded strings are the *raw* model outputs, not the projected env actions, so they go
        through the same ``projection_f`` as during the original rollout and project identically.
        """
        out = list(text_actions)
        for i in np.nonzero(self.replaying(step))[0]:
            out[int(i)] = self.prefix_actions[int(i)][int(step)]
        return out

    def note_dones(self, step: int, dones) -> None:
        """Count replay steps that terminated the episode, which the purity claim forbids."""
        self.early_done += int((self.replaying(step) & np.asarray(dones, dtype=bool)).sum())

    # -- reporting ----------------------------------------------------------------------------
    def plan_stats(self) -> dict[str, float]:
        rows = np.nonzero(self.active)[0]
        return {
            "rewind/degenerate_group_frac": float(self.n_degenerate_groups / max(self.n_groups, 1)),
            "rewind/branched_group_frac": float(self.n_branched_groups / max(self.n_groups, 1)),
            "rewind/n_branch_rows": float(len(rows)),
            "rewind/prefix_len_mean": float(self.prefix_len[rows].mean()) if len(rows) else float("nan"),
            "rewind/depth": float(self.depth),
        }

    def outcome_stats(self, episode_rewards) -> dict[str, float]:
        """The go/no-go numbers: did the branches actually produce outcome variance?

        ``branch_group_informative_frac`` is the one to watch. It is the fraction of branch groups
        whose members did not all end with the same win/lose outcome, i.e. the fraction of the extra
        compute that turned into a non-zero advantage. If it stays near 0 the fork is in the wrong
        place (raise ``depth``) or the suffix is not where the outcome is decided, and no amount of
        estimator tuning will help.
        """
        won = np.asarray(episode_rewards, dtype=np.float64) > 0.0
        by_group: dict[str, list[bool]] = defaultdict(list)
        for i in np.nonzero(self.active)[0]:
            by_group[self.uid[int(i)]].append(bool(won[int(i)]))
        informative = [g for g in by_group.values() if any(g) and not all(g)]
        rows = np.nonzero(self.active)[0]
        return {
            "rewind/n_branch_groups": float(len(by_group)),
            "rewind/branch_group_informative_frac": float(len(informative) / max(len(by_group), 1)),
            "rewind/sr_branch": float(won[rows].mean()) if len(rows) else float("nan"),
            "rewind/replay_early_done": float(self.early_done),
        }

    def degenerate_branch_uids(self, episode_rewards) -> set:
        """Branch groups that came out degenerate anyway -- the input to the next, deeper round."""
        won = np.asarray(episode_rewards, dtype=np.float64) > 0.0
        by_group: dict[str, list[bool]] = defaultdict(list)
        for i in np.nonzero(self.active)[0]:
            by_group[self.uid[int(i)]].append(bool(won[int(i)]))
        return {k for k, v in by_group.items() if all(v) or not any(v)}


def plan_rewind(group_n, goal_idx, won, dense, lengths, actions, n_prefix: int = 4,
                n_branch: int = 2, depth: int = 1, only_all_lose: bool = True,
                restrict_rows=None) -> RewindPlan:
    """Assign every row slot either a (source prefix, branch) job or nothing.

    Args:
        group_n: rollout group size; row ``i`` belongs to group ``i // group_n``.
        goal_idx: per-row env goal index. Constant within a group (``webshop/envs.py:611-612``
            repeats one draw ``group_n`` times), and it is what ``reset_to`` needs to put the env
            back on the same task before replaying.
        won: per-trajectory terminal success.
        dense: per-trajectory dense ``task_score`` -- used only to rank which failures are worth
            re-sampling, never as a reward.
        lengths: per-trajectory number of executed steps.
        actions: per-trajectory list of raw model outputs, in step order.
        n_prefix, n_branch: prefixes per degenerate group and continuations per prefix. Their
            product may not exceed ``group_n``, because a prefix may only be replayed on a worker
            of its own group (shared seed).
        depth: how many trailing steps to discard before re-sampling.
        only_all_lose: branch only all-failed groups. All-won groups are degenerate too, but
            re-sampling them can only manufacture failures, which is not the missing signal.
        restrict_rows: optional set of rows eligible as prefix sources (used by later rounds to
            re-fork only the branch groups that stayed degenerate).
    """
    n = len(won)
    group_n = int(group_n)
    if group_n < 2 or n % group_n != 0:
        raise ValueError(f"CRB needs group_n>=2 dividing the rollout batch, got {n}/{group_n}")
    if int(n_prefix) * int(n_branch) > group_n:
        raise ValueError(
            f"env.rollout.rewind: n_prefix*n_branch={n_prefix}*{n_branch} exceeds group_n={group_n}; "
            "a prefix can only be replayed on a worker of its own group (shared seed), so there are "
            "only group_n slots per group."
        )
    if int(n_branch) < 2:
        raise ValueError("env.rollout.rewind.n_branch must be >=2, or a branch group has no contrast")

    won = np.asarray(won, dtype=bool)
    dense = np.asarray(dense, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    n_groups = n // group_n

    goal_idxs = np.zeros(n, dtype=np.int64)
    active = np.zeros(n, dtype=bool)
    # Idle slots get a throwaway uid rather than a shared one: their rows are dropped before
    # training, but an accidental collision would silently merge two real groups.
    uid = np.array([str(uuid.uuid4()) for _ in range(n)], dtype=object)
    src_row = np.full(n, -1, dtype=np.int64)
    prefix_actions: list[list[str]] = [[] for _ in range(n)]

    n_degenerate = 0
    n_branched = 0
    for g in range(n_groups):
        rows = np.arange(g * group_n, (g + 1) * group_n)
        goal_idxs[rows] = int(goal_idx[rows[0]])
        w = won[rows]
        if bool(w.all()) or not bool(w.any()):
            n_degenerate += 1
        eligible = (not bool(w.any())) if only_all_lose else (bool(w.all()) or not bool(w.any()))
        if not eligible:
            continue
        cand = [
            int(i) for i in rows
            if lengths[i] >= 2 and len(actions[i]) >= 2
            and (restrict_rows is None or int(i) in restrict_rows)
        ]
        if not cand:
            continue
        # Best dense progress first, shorter trajectory as the tie-break: a shorter route to the
        # same score leaves more of the step budget for the re-sampled suffix.
        cand.sort(key=lambda i: (-float(dense[i]), int(lengths[i])))
        slot = 0
        for s in cand[: int(n_prefix)]:
            avail = min(int(lengths[s]), len(actions[s]))
            prefix_len = max(1, avail - int(depth))
            branch_uid = str(uuid.uuid4())
            for _ in range(int(n_branch)):
                r = int(rows[slot])
                slot += 1
                active[r] = True
                uid[r] = branch_uid
                src_row[r] = s
                prefix_actions[r] = list(actions[s][:prefix_len])
        if slot:
            n_branched += 1

    return RewindPlan(
        batch_size=n, group_n=group_n, goal_idxs=goal_idxs, active=active, uid=uid,
        src_row=src_row, prefix_actions=prefix_actions, depth=int(depth),
        n_degenerate_groups=n_degenerate, n_branched_groups=n_branched,
    )


def summarize_trajectories(total_batch_list, episode_rewards):
    """Per-trajectory (goal_idx, won, dense score, length, raw action list) from a finished rollout.

    Only rows with ``active_masks`` count: ``gather_rollout_data`` drops the rest, and the steps
    after termination are padding whose actions were never executed.
    """
    n = len(total_batch_list)
    goal_idx = np.zeros(n, dtype=np.int64)
    dense = np.zeros(n, dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)
    actions: list[list[str]] = []
    for i, traj in enumerate(total_batch_list):
        acts: list[str] = []
        best = 0.0
        gid = -1
        for row in traj:
            if not row["active_masks"]:
                continue
            acts.append(str(row["text_actions"]))
            best = max(best, float(row.get("task_score", 0.0) or 0.0))
            gid = int(row.get("goal_idx", -1))
        actions.append(acts)
        lengths[i] = len(acts)
        dense[i] = best
        goal_idx[i] = gid
    won = np.asarray(episode_rewards, dtype=np.float64) > 0.0
    if (goal_idx < 0).any():
        raise ValueError(
            "env.rollout.rewind.enable needs a per-row 'goal_idx' to replay against, but some "
            "trajectories carry none. Only WebShop provides it today (webshop/envs.py:455)."
        )
    return goal_idx, won, dense, lengths, actions
