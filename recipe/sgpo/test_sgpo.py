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
"""Tests for SGPO (recipe/sgpo/core_sgpo.py).

The first test is load-bearing: lambda=1 must equal GiGPO's step term element for element on a
realistic batch, measured against gigpo's own code. That is what makes "SGPO >= GiGPO" structural --
GiGPO is a member of the family the cross-validation searches over. The rest pin the certainty-
equivalence properties (reward-free groups are identically zero, pure-chain residuals vanish at
lambda=0, duplicated rows inherit their original's advantage).
"""
import numpy as np
import torch

from gigpo.core_gigpo import build_step_group, episode_norm_reward, step_norm_reward
from recipe.sgpo.core_sgpo import (
    ABSORB,
    LAMBDA_GRID,
    build_group_graph,
    certainty_equivalence_values,
    compute_sgpo_outcome_advantage,
    compute_sgpo_step_advantage,
    loto_lambda_curve,
    td_lambda_returns,
)

EPS = 1e-6
GAMMA = 0.95


def _random_batch(n_groups=16, n_traj=8, max_steps=12, n_anchors=6, seed=0):
    """Few distinct anchors on purpose: nodes must be revisited both within and across trajectories,
    since singleton nodes make every normalization agree trivially."""
    rng = np.random.default_rng(seed)
    anchor_obs, index, traj_index, step_ids, rewards = [], [], [], [], []
    for g in range(n_groups):
        for t in range(n_traj):
            length = int(rng.integers(3, max_steps + 1))
            won = rng.random() < 0.4
            for s in range(length):
                anchor_obs.append(f"g{g}_obs{int(rng.integers(0, n_anchors))}")
                index.append(f"uid{g}")
                traj_index.append(f"uid{g}_traj{t}")
                step_ids.append(s)
                rewards.append(10.0 if (won and s == length - 1) else 0.0)
    return (np.array(anchor_obs, dtype=object), np.array(index, dtype=object),
            np.array(traj_index, dtype=object), np.array(rewards, dtype=np.float64),
            np.array(step_ids), np.ones(len(anchor_obs), dtype=bool))


def _mc(rewards, traj_index, step_ids, gamma=GAMMA):
    """batch['step_rewards'] as gigpo.compute_step_discounted_returns produces it."""
    out = np.zeros(len(rewards))
    for tid in np.unique(traj_index):
        idx = sorted(np.flatnonzero(traj_index == tid), key=lambda i: step_ids[i])
        acc = 0.0
        for i in reversed(idx):
            acc = rewards[i] + gamma * acc
            out[i] = acc
    return out


def test_lambda_one_is_gigpo():
    """lam=1: G^1_i is the row's own discounted return and per-node centering removes V(sigma_i), so
    the step advantage must equal gigpo's step_norm_reward(mode=mean_std_norm) exactly."""
    anchor_obs, index, traj_index, rewards, step_ids, active = _random_batch()
    n = len(anchor_obs)
    step_returns = _mc(rewards, traj_index, step_ids)

    ref = step_norm_reward(
        torch.tensor(step_returns, dtype=torch.float32), torch.ones(n, 1),
        build_step_group(anchor_obs, index, False, GAMMA), EPS, remove_std=False,
    )[:, 0].numpy().astype(np.float64)

    got, _, stats = compute_sgpo_step_advantage(
        anchor_obs, index, traj_index, rewards, step_ids, active, gamma=GAMMA, lam=1.0, epsilon=EPS)

    d = np.abs(ref - got)
    assert stats['lam'] == 1.0
    assert d.max() < 1e-6, f"lam=1 is NOT GiGPO's step term: max abs diff {d.max():.3e}"
    print(f"OK lam=1 == GiGPO mean_std_norm: {n} rows, max abs diff {d.max():.3e} "
          f"(|ref| mean {np.abs(ref).mean():.4f})")


def test_dp_w_zero_matches_episode_only():
    bs, resp_len = 4, 6
    token_level_rewards = torch.zeros(bs, resp_len)
    token_level_rewards[:, -1] = torch.tensor([10.0, 10.0, 0.0, 0.0])
    response_mask = torch.ones(bs, resp_len)
    index = np.array(['g0'] * 4, dtype=object)
    traj_index = np.array(['t0', 't1', 't2', 't3'], dtype=object)
    anchor_obs = np.array(['s0', 's0', 's0', 's1'], dtype=object)
    rewards = np.array([10.0, 10.0, 0.0, 0.0])
    step_ids = np.array([0, 0, 0, 0])
    active = np.array([True] * 4)

    expected = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, EPS)
    got, _ = compute_sgpo_outcome_advantage(
        token_level_rewards=token_level_rewards, response_mask=response_mask, anchor_obs=anchor_obs,
        index=index, traj_index=traj_index, rewards=rewards, step_ids=step_ids, active_masks=active,
        gamma=GAMMA, dp_w=0.0)
    err = (got - expected).abs().max().item()
    assert err < 1e-6, err
    print(f"OK dp_w=0 == episode_norm_reward, max|delta| {err:.3e}")


def test_value_is_expectation_not_max():
    """Four trajectories leave s0 for a modest but frequent successor and one leaves for a rare,
    better one. Certainty equivalence returns the frequency-weighted expectation, strictly inside the
    range of the two edges rather than the best edge."""
    anchor, traj, rew, steps = [], [], [], []
    for k in range(4):
        anchor += ['s0', 's1']; traj += [f'thick{k}'] * 2; rew += [0.0, 10.0]; steps += [0, 1]
    anchor += ['s0', 's2']; traj += ['thin0'] * 2; rew += [0.0, 20.0]; steps += [0, 1]
    anchor_obs = np.array(anchor, dtype=object)
    n = len(anchor)
    graphs = build_group_graph(anchor_obs, np.array(['g0'] * n, dtype=object),
                              np.array(traj, dtype=object), np.array(rew), np.array(steps),
                              np.ones(n, dtype=bool))
    V = certainty_equivalence_values(graphs['g0']['edges'], GAMMA)

    assert abs(V['s1'] - 10.0) < 1e-9, V         # s1 -> ABSORB with r=10, seen 4x
    assert abs(V['s2'] - 20.0) < 1e-9, V         # s2 -> ABSORB with r=20, seen 1x
    expected = GAMMA * (0.8 * 10.0 + 0.2 * 20.0)  # 4/5 of s0's transitions go to s1
    assert abs(V['s0'] - expected) < 1e-9, (V['s0'], expected)
    assert V['s0'] < GAMMA * V['s2'], "expectation must stay strictly below the max edge"
    print(f"OK V(s0)={V['s0']:.4f} is the frequency-weighted expectation "
          f"(max edge would be {GAMMA * V['s2']:.4f})")


def test_chain_row_residual_is_exactly_zero():
    """A node with a single out-edge observed once satisfies V(sigma) = r + gamma*V(sigma') by
    construction, so its lambda=0 residual is exactly 0 and the row excludes itself from the centering
    statistics -- no uninformative-node rule is needed."""
    anchor_obs = np.array(['a0', 'a1', 'a2'], dtype=object)
    n = 3
    graphs = build_group_graph(anchor_obs, np.array(['g0'] * n, dtype=object),
                              np.array(['t0'] * n, dtype=object), np.array([0.0, 0.0, 7.0]),
                              np.array([0, 1, 2]), np.ones(n, dtype=bool))
    g = graphs['g0']
    V = certainty_equivalence_values(g['edges'], GAMMA)
    G = td_lambda_returns(g['trajs'], g['node_of'], np.array([0.0, 0.0, 7.0]), V, GAMMA, 0.0)
    res = [G[i] - V[g['node_of'][i]] for i in range(n)]
    assert max(abs(r) for r in res) < 1e-12, res
    print(f"OK pure-chain residuals are exactly 0 at lam=0: max|res| {max(abs(r) for r in res):.1e}")


def test_reward_free_group_is_identically_zero():
    """A group where nothing was rewarded has V == 0 and G == 0 at every lambda, so its advantage is
    exactly 0 -- no floor or fallback is needed to suppress it."""
    anchor_obs, index, traj_index, rewards, step_ids, active = _random_batch(n_groups=3, seed=7)
    rewards = np.zeros_like(rewards)
    for lam in (None, 0.0, 0.5, 1.0):
        adv, raw, stats = compute_sgpo_step_advantage(
            anchor_obs, index, traj_index, rewards, step_ids, active, gamma=GAMMA, lam=lam)
        assert np.abs(adv).max() == 0.0, (lam, np.abs(adv).max())
        assert np.abs(raw).max() == 0.0, (lam, np.abs(raw).max())
        assert stats['frac_groups_zeroed'] == 1.0, stats
    print("OK reward-free groups are identically zero at every lambda, with no gate")


def test_stitching_propagates_value_across_trajectories():
    """t0 walks s0 -> s1 and then runs out of steps, never seeing a reward. t1 enters at s1 and reaches
    a reward from there. Neither trajectory's own MC return from s0 sees the win; the stitched graph
    does, and that is the entire reason for building the graph in the first place."""
    anchor_obs = np.array(['s0', 's1', 's1', 's2'], dtype=object)
    rewards = np.array([0.0, 0.0, 0.0, 10.0])
    n = 4
    graphs = build_group_graph(anchor_obs, np.array(['g0'] * n, dtype=object),
                              np.array(['t0', 't0', 't1', 't1'], dtype=object), rewards,
                              np.array([0, 1, 0, 1]), np.ones(n, dtype=bool))
    V = certainty_equivalence_values(graphs['g0']['edges'], GAMMA)
    # s1 has two observed out-edges (t0's to ABSORB, t1's to s2), so its value is the average of
    # dying and winning -- not the max, and not 0 as unstitched MC would have it.
    assert abs(V['s2'] - 10.0) < 1e-9, V
    assert abs(V['s1'] - 0.5 * GAMMA * 10.0) < 1e-9, V
    assert abs(V['s0'] - GAMMA * V['s1']) < 1e-9, V
    assert V['s0'] > 0.0
    print(f"OK stitched V(s0)={V['s0']:.4f} > 0 while t0's own MC return from s0 is 0")


def test_loto_curve_is_well_formed():
    anchor_obs, index, traj_index, rewards, step_ids, active = _random_batch()
    graphs = build_group_graph(anchor_obs, index, traj_index, rewards, step_ids, active)
    sse, n_scored, coverage = loto_lambda_curve(graphs, GAMMA)
    assert sse.shape == (len(LAMBDA_GRID),)
    assert np.isfinite(sse).all()
    assert n_scored > 0 and 0.0 < coverage <= 1.0
    adv, _, stats = compute_sgpo_step_advantage(
        anchor_obs, index, traj_index, rewards, step_ids, active, gamma=GAMMA, lam=None)
    assert stats['lam'] in LAMBDA_GRID
    assert abs(stats['loto_coverage'] - coverage) < 1e-12
    print(f"OK LOTO curve {np.round(sse / n_scored, 3).tolist()} over lam={list(LAMBDA_GRID)} "
          f"-> selected lam={stats['lam']}, coverage {coverage:.3f}, {n_scored} scored rows")


def test_duplicated_rows_do_not_change_the_graph():
    """adjust_batch(mode='copy') pads the batch with exact row copies *before* compute_advantage runs
    in this trainer, so a duplicate shares its original's (traj_uid, step_id). It must not become a
    sigma -> sigma self-edge or a second vote in Phat, and it must receive its original's advantage."""
    anchor_obs, index, traj_index, rewards, step_ids, active = _random_batch(n_groups=4, seed=11)
    base, _, base_stats = compute_sgpo_step_advantage(
        anchor_obs, index, traj_index, rewards, step_ids, active, gamma=GAMMA, lam=0.5)

    rng = np.random.default_rng(5)
    dup = rng.choice(len(anchor_obs), 40, replace=False)
    cat = lambda a: np.concatenate([a, a[dup]])                                  # noqa: E731
    got, _, stats = compute_sgpo_step_advantage(
        cat(anchor_obs), cat(index), cat(traj_index), cat(rewards), cat(step_ids), cat(active),
        gamma=GAMMA, lam=0.5)

    assert stats['n_dup_rows'] == 40, stats
    assert stats['n_nodes_total'] == base_stats['n_nodes_total'], (stats, base_stats)
    assert np.abs(got[:len(base)] - base).max() < 1e-12
    assert np.abs(got[len(base):] - base[dup]).max() < 1e-12
    print(f"OK {len(dup)} duplicated rows leave the graph identical and inherit their originals' "
          f"advantage")


if __name__ == '__main__':
    test_lambda_one_is_gigpo()
    test_dp_w_zero_matches_episode_only()
    test_value_is_expectation_not_max()
    test_chain_row_residual_is_exactly_zero()
    test_reward_free_group_is_identically_zero()
    test_stitching_propagates_value_across_trajectories()
    test_loto_curve_is_well_formed()
    test_duplicated_rows_do_not_change_the_graph()
    print("all SGPO tests passed")
