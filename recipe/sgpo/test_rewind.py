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
"""Tests for CRB's rewind planner (recipe/sgpo/rewind.py).

A) Only all-failed groups are branched; a group containing a winner is left alone. Slot geometry,
   prefix contents and the fresh-uid-per-prefix rule.
B) Source ranking: the highest dense score wins, shorter trajectory breaks a tie.
C) The per-step protocol -- which rows are replaying, and that script_actions rewrites exactly those.
D) outcome_stats / degenerate_branch_uids on hand-built outcomes.
E) summarize_trajectories reads lengths, actions and dense scores off active rows only.
F) Configuration errors that must raise rather than silently under-branch.
"""
import numpy as np

from recipe.sgpo.rewind import plan_rewind, summarize_trajectories


def _acts(tag, n):
    return [f"{tag}_a{k}" for k in range(n)]


def test_a_only_all_lose_groups_are_branched():
    group_n = 4
    # group 0: all four failed -> branchable. group 1: row 5 won -> left alone.
    won = np.array([False] * 4 + [False, True, False, False])
    dense = np.array([0.1, 0.9, 0.5, 0.3, 0.2, 1.0, 0.2, 0.2])
    lengths = np.array([3, 4, 5, 6, 3, 3, 3, 3])
    actions = [_acts(f"t{i}", int(lengths[i])) for i in range(8)]
    goal_idx = np.array([7, 7, 7, 7, 9, 9, 9, 9])

    plan = plan_rewind(group_n=group_n, goal_idx=goal_idx, won=won, dense=dense, lengths=lengths,
                       actions=actions, n_prefix=2, n_branch=2, depth=1)

    assert plan.active[:4].all(), plan.active
    assert not plan.active[4:].any(), plan.active
    # Both groups' goals are recorded (reset_to needs all of them), taken from the group's first row.
    assert list(plan.goal_idxs) == [7, 7, 7, 7, 9, 9, 9, 9]
    # Two prefixes x two branches: rows 0,1 share one uid and rows 2,3 another, and the two uids
    # differ -- one branch group per prefix, never one per source group.
    assert plan.uid[0] == plan.uid[1] and plan.uid[2] == plan.uid[3]
    assert plan.uid[0] != plan.uid[2]
    # Best dense in group 0 is row 1 (0.9), second is row 2 (0.5).
    assert list(plan.src_row[:4]) == [1, 1, 2, 2]
    # depth=1 -> replay L-1 actions of the source.
    assert plan.prefix_actions[0] == _acts("t1", 4)[:3]
    assert plan.prefix_actions[2] == _acts("t2", 5)[:4]
    assert list(plan.prefix_len[:4]) == [3, 3, 4, 4]
    st = plan.plan_stats()
    assert st["rewind/degenerate_group_frac"] == 0.5
    assert st["rewind/branched_group_frac"] == 0.5
    assert st["rewind/n_branch_rows"] == 4.0
    print(f"A OK: branched rows {np.nonzero(plan.active)[0].tolist()}, sources {plan.src_row[:4].tolist()}")


def test_b_source_ranking_prefers_high_dense_then_short():
    group_n = 4
    won = np.zeros(4, dtype=bool)
    # rows 1 and 3 tie on dense; row 3 is shorter and must be picked first.
    dense = np.array([0.2, 0.7, 0.1, 0.7])
    lengths = np.array([5, 6, 5, 3])
    actions = [_acts(f"t{i}", int(lengths[i])) for i in range(4)]
    plan = plan_rewind(group_n=group_n, goal_idx=np.zeros(4, dtype=int), won=won, dense=dense,
                       lengths=lengths, actions=actions, n_prefix=2, n_branch=2, depth=2)
    assert list(plan.src_row) == [3, 3, 1, 1], plan.src_row
    # depth=2 -> prefix L-2; row 3 has L=3 so prefix length 1.
    assert list(plan.prefix_len) == [1, 1, 4, 4], plan.prefix_len
    print(f"B OK: sources {plan.src_row.tolist()} prefix_len {plan.prefix_len.tolist()}")


def test_b2_prefix_never_empty_on_a_two_step_trajectory():
    """depth may exceed the trajectory length; the fork must still leave at least one replayed
    action, otherwise the 'branch' is plain i.i.d. resampling and the shared-prefix contrast the
    branch group's advantage relies on does not exist."""
    won = np.zeros(2, dtype=bool)
    plan = plan_rewind(group_n=2, goal_idx=np.zeros(2, dtype=int), won=won,
                       dense=np.array([0.5, 0.4]), lengths=np.array([2, 2]),
                       actions=[_acts("t0", 2), _acts("t1", 2)],
                       n_prefix=1, n_branch=2, depth=5)
    assert list(plan.prefix_len) == [1, 1], plan.prefix_len
    print("B2 OK: prefix floored at 1 action")


def test_c_replaying_and_script_actions():
    won = np.zeros(4, dtype=bool)
    dense = np.array([0.9, 0.1, 0.1, 0.1])
    lengths = np.array([4, 3, 3, 3])
    actions = [_acts(f"t{i}", int(lengths[i])) for i in range(4)]
    plan = plan_rewind(group_n=4, goal_idx=np.zeros(4, dtype=int), won=won, dense=dense,
                       lengths=lengths, actions=actions, n_prefix=1, n_branch=2, depth=1)
    # prefix = t0's first 3 actions on rows 0 and 1; rows 2,3 idle.
    assert list(plan.prefix_len) == [3, 3, 0, 0]
    for step in range(3):
        rep = plan.replaying(step)
        assert list(rep) == [True, True, False, False], (step, rep)
        scripted = plan.script_actions(step, ["GEN"] * 4)
        assert scripted[0] == scripted[1] == f"t0_a{step}", scripted
        assert scripted[2] == scripted[3] == "GEN", scripted
    # Past the prefix the rows generate for real and nothing is rewritten.
    assert not plan.replaying(3).any()
    assert plan.script_actions(3, ["GEN"] * 4) == ["GEN"] * 4
    print("C OK: 3 scripted steps on both branches, then free sampling")


def test_d_outcome_stats_and_next_round_sources():
    won = np.zeros(8, dtype=bool)
    dense = np.array([0.9, 0.8, 0.1, 0.1, 0.9, 0.8, 0.1, 0.1])
    lengths = np.full(8, 4)
    actions = [_acts(f"t{i}", 4) for i in range(8)]
    plan = plan_rewind(group_n=4, goal_idx=np.array([1, 1, 1, 1, 2, 2, 2, 2]), won=won, dense=dense,
                       lengths=lengths, actions=actions, n_prefix=2, n_branch=2, depth=1)
    assert plan.active.all()
    # Four branch groups (two per source group). Make exactly one of them informative: rows 0,1 are
    # one branch group, so give row 0 a win and row 1 a loss.
    er = np.zeros(8)
    er[0] = 10.0
    st = plan.outcome_stats(er)
    assert st["rewind/n_branch_groups"] == 4.0, st
    assert abs(st["rewind/branch_group_informative_frac"] - 0.25) < 1e-9, st
    assert abs(st["rewind/sr_branch"] - 0.125) < 1e-9, st
    # The three flat branch groups are what a second, deeper round would re-fork.
    flat = plan.degenerate_branch_uids(er)
    assert len(flat) == 3, flat
    assert plan.uid[0] not in flat
    print(f"D OK: informative_frac {st['rewind/branch_group_informative_frac']}, "
          f"{len(flat)} flat branch groups left for the next round")


def test_e_summarize_trajectories_reads_active_rows_only():
    def row(action, score, active, goal=5):
        return {'text_actions': action, 'task_score': score, 'active_masks': active, 'goal_idx': goal}

    total_batch_list = [
        [row('a0', 0.0, True), row('a1', 0.42, True), row('pad', 0.0, False)],
        [row('b0', 0.0, True), row('b1', 0.0, True), row('b2', 1.0, True)],
    ]
    episode_rewards = np.array([0.0, 10.0])
    goal_idx, won, dense, lengths, actions = summarize_trajectories(total_batch_list, episode_rewards)
    assert list(lengths) == [2, 3], lengths
    assert actions[0] == ['a0', 'a1'], actions[0]
    assert abs(dense[0] - 0.42) < 1e-9 and abs(dense[1] - 1.0) < 1e-9
    assert list(won) == [False, True]
    assert list(goal_idx) == [5, 5]
    print("E OK: padded rows excluded, dense = max task_score over active rows")


def test_f_config_errors_raise():
    kw = dict(goal_idx=np.zeros(4, dtype=int), won=np.zeros(4, dtype=bool),
              dense=np.zeros(4), lengths=np.full(4, 3),
              actions=[_acts(f"t{i}", 3) for i in range(4)])
    for bad, why in (
        (dict(n_prefix=3, n_branch=2), "n_prefix*n_branch must fit in group_n"),
        (dict(n_prefix=4, n_branch=1), "n_branch<2 leaves a branch group with no contrast"),
    ):
        try:
            plan_rewind(group_n=4, depth=1, **kw, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {why}")
    try:
        plan_rewind(group_n=3, depth=1, **kw, n_prefix=1, n_branch=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError: group_n must divide the batch")
    print("F OK: all three misconfigurations raise")


if __name__ == '__main__':
    test_a_only_all_lose_groups_are_branched()
    test_b_source_ranking_prefers_high_dense_then_short()
    test_b2_prefix_never_empty_on_a_two_step_trajectory()
    test_c_replaying_and_script_actions()
    test_d_outcome_stats_and_next_round_sources()
    test_e_summarize_trajectories_reads_active_rows_only()
    test_f_config_errors_raise()
    print("all CRB rewind tests passed")
