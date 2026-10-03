"""SGPO: a state-graph step estimator (algorithm.adv_estimator=sgpo).

A group's rollouts share one env instance, so they are stitched into one state graph and value is
propagated through it. The estimator is the certainty-equivalence (batch-TD(0) fixed point) value of
the group's empirical MDP, with a TD(lambda) return per row on top:

    V(sigma)   = rhat(sigma) + gamma * sum_{sigma'} Phat(sigma'|sigma) V(sigma'),  V(ABSORB) = 0
    G^lambda_i = r_i + gamma * [ (1 - lambda) V(sigma_{i+1}) + lambda * G^lambda_{i+1} ]
    A_i        = G^lambda_i - V(sigma_i),   standardized within each anchor state.

The episode-level term is unchanged (episode_norm_reward from core_gigpo).

lambda = 1 is GiGPO exactly: G^1_i is the row's discounted return and per-anchor standardization
cancels V(sigma_i), so the step term equals step_norm_reward(mode=mean_std_norm). lambda is
cross-validated per batch (leave-one-trajectory-out over LAMBDA_GRID), not tuned.

Terminal rows are detected via step_id + active_masks (no `dones` field exists): a trajectory's last
surviving row is the one whose successor is ABSORB.
"""
import hashlib

import numpy as np
import torch
from collections import defaultdict

from gigpo.core_gigpo import episode_norm_reward


ABSORB = '__sgpo_done__'

# Grid for the cross-validated lambda search: 1.0 is GiGPO, 0.0 is pure graph bootstrapping.
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _per_node_standardize(nodes, values: np.ndarray, eps: float, no_std: bool = False) -> np.ndarray:
    """Center and scale `values` within each anchor-state group (GiGPO Eq.7's normalization).

    Per-node mean, per-node unbiased (ddof=1) std with std:=1 for a singleton (so it gets exactly 0),
    eps added to the std. `nodes[k]` keys `values[k]`. Matching GiGPO's normalization is what makes
    lambda the only difference between the two methods.

    no_std=True is the "wo std norm" ablation: drop the std division, center only. Centering still
    cancels V(sigma_i), so lambda remains the single difference between GiGPO and SGPO.
    """
    buckets = defaultdict(list)
    for k, node in enumerate(nodes):
        buckets[node].append(k)
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    for ks in buckets.values():
        ks = np.asarray(ks)
        v = values[ks]
        if no_std:
            out[ks] = v - v.mean()
        else:
            std = 1.0 if v.size == 1 else float(np.std(v, ddof=1))
            out[ks] = (v - v.mean()) / (std + eps)
    return out


def build_group_graph(anchor_obs, index, traj_index, rewards, step_ids, active_masks):
    """One graph per episode-group index.

    Nodes are anchor (sigma) strings; a terminal row's successor is ABSORB (whether the trajectory
    ended by done or by exhausting max_steps -- in both cases there is no observed successor row).

    Returns group_id -> {
        'edges': {node: {next_node: [reward_sum, count]}},  # observed transitions, keyed by actual
                  # successor so distinct successors of one sigma stay distinct out-edges.
        'trajs': [[row_idx, ...], ...],   # per trajectory, rows in step order.
    }
    """
    n = len(anchor_obs)
    by_traj = defaultdict(list)
    for i in range(n):
        if active_masks is not None and not active_masks[i]:
            continue
        by_traj[traj_index[i]].append(i)

    graphs = defaultdict(lambda: {'trajs': []})
    for tid, idxs in by_traj.items():
        idxs = sorted(idxs, key=lambda i: step_ids[i])  # sort defensively by step order
        graphs[index[idxs[0]]]['trajs'].append(idxs)

    node_of = {i: str(anchor_obs[i]) for i in range(n)}
    for g in graphs.values():
        g['edges'] = _edges_of(g['trajs'], node_of, rewards)
        g['node_of'] = node_of
        g['rewards'] = rewards
    return graphs


def _edges_of(trajs, node_of, rewards):
    """{node: {next_node: [reward_sum, count]}} over the given list of trajectories' row lists."""
    edges = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for rows in trajs:
        for pos, i in enumerate(rows):
            nxt = ABSORB if pos + 1 == len(rows) else node_of[rows[pos + 1]]
            slot = edges[node_of[i]][nxt]
            slot[0] += float(rewards[i])
            slot[1] += 1
    return edges


def _group_shuffle_rng(shuffle_seed: int, group_id) -> np.random.Generator:
    """Per-group RNG for the Shuffled-CE ablation, seeded only by (shuffle_seed, group_id).

    Independent of the rollout RNG and order-independent across groups. group_id is hashed to a
    stable 32-bit value (Python's salted hash() is not reproducible across processes).
    """
    h = int(hashlib.blake2b(str(group_id).encode(), digest_size=8).hexdigest(), 16)
    return np.random.default_rng([int(shuffle_seed) & 0xFFFFFFFF, h & 0xFFFFFFFF])


def _shuffle_successor_labels(edges, rng: np.random.Generator):
    """Shuffled-CE ablation: permute the non-terminal successor labels of one group's fitted graph.

    A random bijection relabels every live out-edge target; source nodes, reward_sum/count and ABSORB
    edges are untouched, and colliding edges are merged by summing reward_sum and count. Groups with
    fewer than two distinct live successors are returned unchanged. Only the CE value solve consumes
    this graph. Returns (edges_out, n_edges_relabelled, n_live_edges).
    """
    succ = sorted({nxt for slots in edges.values() for nxt in slots if nxt != ABSORB})
    n_live = sum(1 for slots in edges.values() for nxt in slots if nxt != ABSORB)
    if len(succ) < 2:
        return edges, 0, n_live
    perm = rng.permutation(len(succ))
    pi = {succ[k]: succ[perm[k]] for k in range(len(succ))}
    out = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    n_changed = 0
    for s, slots in edges.items():
        for nxt, (r_sum, cnt) in slots.items():
            mapped = nxt if nxt == ABSORB else pi[nxt]
            slot = out[s][mapped]
            slot[0] += r_sum
            slot[1] += cnt
            if mapped != nxt:
                n_changed += cnt
    return out, n_changed, n_live


def certainty_equivalence_values(edges, gamma: float):
    """V = rhat + gamma * Phat V with V(ABSORB) = 0, solved exactly.

    The value of the empirical MDP built from the group's pooled transitions (the batch-TD(0) fixed
    point). Phat is substochastic on the live nodes, so I - gamma*Phat is strictly diagonally
    dominant for gamma < 1 and the direct solve is well conditioned. gamma >= 1 is rejected: the
    system is then singular for any recurrent group.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"sgpo requires 0 <= gamma < 1 for the certainty-equivalence solve, got {gamma}")
    nodes = [s for s in edges if s != ABSORB]
    if not nodes:
        return {ABSORB: 0.0}
    idx = {s: k for k, s in enumerate(nodes)}
    A = np.eye(len(nodes), dtype=np.float64)
    b = np.zeros(len(nodes), dtype=np.float64)
    for s, slots in edges.items():
        if s == ABSORB:
            continue
        k = idx[s]
        tot = float(sum(cnt for _, cnt in slots.values()))
        for nxt, (r_sum, cnt) in slots.items():
            b[k] += r_sum / tot
            if nxt in idx:
                A[k, idx[nxt]] -= gamma * cnt / tot
    V = dict(zip(nodes, np.linalg.solve(A, b)))
    V[ABSORB] = 0.0
    return V


def td_lambda_returns(trajs, node_of, rewards, V, gamma: float, lam: float):
    """{row_idx: G^lambda}. Backward recursion inside each trajectory; V(ABSORB)=0 and G past the
    last row is 0, so a terminal row's return is just its own reward and lambda=1 reproduces the
    plain discounted return GiGPO's step term uses."""
    out = {}
    for rows in trajs:
        g_next = 0.0
        v_next = 0.0
        for pos in range(len(rows) - 1, -1, -1):
            i = rows[pos]
            g = float(rewards[i]) + gamma * ((1.0 - lam) * v_next + lam * g_next)
            out[i] = g
            g_next = g
            v_next = V.get(node_of[i], 0.0)
    return out


def _mc_returns(trajs, rewards, gamma: float):
    """{row_idx: own discounted return}, the LOTO target. Recomputed locally so selection does not
    depend on a batch tensor that may have been padded or duplicated upstream."""
    out = {}
    for rows in trajs:
        acc = 0.0
        for i in reversed(rows):
            acc = float(rewards[i]) + gamma * acc
            out[i] = acc
    return out


def loto_lambda_curve(graphs, gamma: float, lambdas=LAMBDA_GRID):
    """Pooled leave-one-trajectory-out squared error of Vtilde^lambda, per lambda.

    For each group and each rollout h: build V from the other rollouts, average G^lambda over the fit
    rows at each node to get Vtilde^lambda(node), then score every row of h whose node the fit rows
    also visited against that row's own realized discounted return. Coverage is a function of the node
    partition only, identical across lambda. Returns (sse_per_lambda, n_scored, coverage).
    """
    sse = np.zeros(len(lambdas), dtype=np.float64)
    n_scored = 0
    n_rows = 0
    for g in graphs.values():
        trajs, node_of = g['trajs'], g['node_of']
        if len(trajs) < 2:
            continue
        mc = _mc_returns(trajs, g['rewards'], gamma)
        for h in range(len(trajs)):
            fit = [rows for k, rows in enumerate(trajs) if k != h]
            edges = _edges_of(fit, node_of, g['rewards'])
            V = certainty_equivalence_values(edges, gamma)
            fit_rows_of_node = defaultdict(list)
            for rows in fit:
                for i in rows:
                    fit_rows_of_node[node_of[i]].append(i)
            held = [i for i in trajs[h] if node_of[i] in fit_rows_of_node]
            n_rows += len(trajs[h])
            n_scored += len(held)
            if not held:
                continue
            target = np.array([mc[i] for i in held])
            for li, lam in enumerate(lambdas):
                G = td_lambda_returns(fit, node_of, g['rewards'], V, gamma, lam)
                vt = {s: float(np.mean([G[i] for i in ii])) for s, ii in fit_rows_of_node.items()}
                pred = np.array([vt[node_of[i]] for i in held])
                sse[li] += float(np.sum((pred - target) ** 2))
    coverage = n_scored / max(n_rows, 1)
    return sse, n_scored, coverage


def _dedup_active(traj_index, step_ids, active_masks):
    """(evidence mask, {duplicate row -> canonical row}).

    `adjust_batch(mode='copy')` pads the batch by appending exact row copies before compute_advantage.
    A duplicate shares its original's (traj_uid, step_id), so left alone it would form a spurious
    sigma -> sigma self-edge and count twice in Phat. Duplicates are excluded from the graph and given
    their original's advantage afterwards.
    """
    seen = {}
    keep = np.zeros(len(traj_index), dtype=bool)
    alias = {}
    for i in range(len(traj_index)):
        if active_masks is not None and not active_masks[i]:
            continue
        key = (traj_index[i], step_ids[i])
        if key in seen:
            alias[i] = seen[key]
        else:
            seen[key] = i
            keep[i] = True
    return keep, alias


def compute_sgpo_step_advantage(anchor_obs, index, traj_index, rewards, step_ids, active_masks,
                                 gamma: float, lam: float = None, epsilon: float = 1e-6,
                                 lambdas=LAMBDA_GRID, no_std: bool = False,
                                 shuffle_ce: bool = False, shuffle_seed: int = 0):
    """Returns (per-row advantage, per-row raw TD(lambda) residual, stats).

    A_i = G^lambda_i - V(sigma_i), standardized within each anchor state. lam=None selects lambda by
    pooled leave-one-trajectory-out cross-validation over `lambdas`; a float fixes it (lam=1.0 is
    GiGPO's step term). Per-node centering cancels V(sigma_i), leaving a comparison between the
    successors actually taken from that state.

    shuffle_ce=True is the Shuffled-CE ablation: V is solved on a graph with permuted non-terminal
    successor labels, while the TD(lambda) recursion, anchor lookups and centering stay on the real
    trajectories. Only the CE value function changes.
    """
    n = len(anchor_obs)
    adv = np.zeros(n, dtype=np.float64)
    raw = np.zeros(n, dtype=np.float64)
    evidence, alias = _dedup_active(traj_index, step_ids, active_masks)
    graphs = build_group_graph(anchor_obs, index, traj_index, rewards, step_ids, evidence)

    curve = None
    if lam is None:
        sse, n_scored, coverage = loto_lambda_curve(graphs, gamma, lambdas)
        lam = float(lambdas[int(np.argmin(sse))]) if n_scored else 1.0
        curve = sse / max(n_scored, 1)
    else:
        lam, coverage = float(lam), float('nan')

    n_nodes_total = n_active = n_signal = 0
    n_edges = n_edges_multi = out_degree_sum = n_groups_zeroed = 0
    n_edges_shuffled = n_edges_live = n_groups_shuffled = 0
    for gid, g in graphs.items():
        node_of = g['node_of']
        if shuffle_ce:
            edges_for_V, n_chg, n_live = _shuffle_successor_labels(
                g['edges'], _group_shuffle_rng(shuffle_seed, gid))
            n_edges_shuffled += n_chg
            n_edges_live += n_live
            n_groups_shuffled += int(n_chg > 0)
        else:
            edges_for_V = g['edges']
        V = certainty_equivalence_values(edges_for_V, gamma)
        G = td_lambda_returns(g['trajs'], node_of, rewards, V, gamma, lam)
        n_nodes_total += len(V)
        for slots in g['edges'].values():
            out_degree_sum += len(slots)
            n_edges += len(slots)
            n_edges_multi += sum(1 for _, (_, cnt) in slots.items() if cnt > 1)

        rows = [i for t in g['trajs'] for i in t]
        residual = np.array([G[i] - V[node_of[i]] for i in rows])
        centered = _per_node_standardize([node_of[i] for i in rows], residual, epsilon, no_std=no_std)
        n_active += len(rows)
        n_signal += int((np.abs(centered) > 1e-9).sum())
        n_groups_zeroed += int(np.abs(centered).max(initial=0.0) <= 1e-9)
        for k, i in enumerate(rows):
            adv[i] = centered[k]
            raw[i] = residual[k]
    for dup, src in alias.items():
        adv[dup] = adv[src]
        raw[dup] = raw[src]

    n_g = len(graphs)
    stats = {
        'lam': float(lam),
        'n_groups': n_g,
        'n_nodes_total': n_nodes_total,
        'nodes_per_group': (n_nodes_total / n_g) if n_g else 0.0,
        'mean_out_degree': (out_degree_sum / max(n_nodes_total - n_g, 1)),
        'frac_edges_multi': (n_edges_multi / n_edges) if n_edges else 0.0,
        # All-zero exactly when nothing in the group was rewarded (a property of the batch, not a
        # gate); GiGPO zeroes the same groups.
        'frac_groups_zeroed': (n_groups_zeroed / n_g) if n_g else 0.0,
        'signal_frac': (n_signal / n_active) if n_active else 0.0,
        'loto_coverage': coverage,
        # Shuffled-CE: how much of the fitted graph the permutation moved (NaN frac when off).
        'shuffle_ce': float(bool(shuffle_ce)),
        'n_groups_shuffled': n_groups_shuffled,
        'frac_edges_shuffled': (n_edges_shuffled / n_edges_live) if (shuffle_ce and n_edges_live) else float('nan'),
        # Rows dropped as adjust_batch copies; >0 means the batch was padded before advantages.
        'n_dup_rows': len(alias),
    }
    if curve is not None:
        # The full cross-validation curve: whether the data preferred graph bootstrapping (lam < 1)
        # over GiGPO (lam = 1) and by what margin.
        for l_, mse in zip(lambdas, curve):
            stats[f'loto_mse/lam{l_:g}'] = float(mse)
    return adv, raw, stats


def compute_sgpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                    response_mask: torch.Tensor,
                                    anchor_obs: np.array,
                                    index: np.array,
                                    traj_index: np.array,
                                    rewards: np.array,
                                    step_ids: np.array,
                                    active_masks: np.array,
                                    gamma: float,
                                    dp_w: float = 1.0,
                                    lam: float = None,
                                    epsilon: float = 1e-6,
                                    no_std: bool = False,
                                    no_episode: bool = False,
                                    shuffle_ce: bool = False,
                                    shuffle_seed: int = 0,
                                    return_stats: bool = False):
    """GiGPO's episode term + dp_w * (state-graph TD(lambda) step term).

    dp_w=0 leaves the episode term alone (plain GRPO over episodes). dp_w=1 with lam=1.0 is GiGPO
    (step_advantage_w=1); dp_w=1 with lam=None is SGPO, cross-validating lambda per batch.

    no_std=True is the "wo std norm" ablation: the step term is centered per node but not scaled, and
    the episode term uses remove_std=True to match. no_episode=True is the "w/o episode" ablation:
    drop the episode term (A = dp_w * A_step). shuffle_ce/shuffle_seed are the Shuffled-CE ablation,
    forwarded to the step term.
    """
    response_length = response_mask.shape[-1]
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index,
                                             epsilon, remove_std=True)
    if no_episode:
        episode_advantages = torch.zeros_like(episode_advantages)

    step_adv_np, step_raw_np, stats = compute_sgpo_step_advantage(
        anchor_obs=anchor_obs, index=index, traj_index=traj_index, rewards=rewards,
        step_ids=step_ids, active_masks=active_masks, gamma=gamma, lam=lam, epsilon=epsilon,
        no_std=no_std, shuffle_ce=shuffle_ce, shuffle_seed=shuffle_seed,
    )
    step_adv = torch.tensor(step_adv_np, dtype=torch.float32, device=token_level_rewards.device)
    step_advantages = step_adv.unsqueeze(-1).tile([1, response_length]) * response_mask
    stats['no_episode'] = float(bool(no_episode))

    scores = episode_advantages + dp_w * step_advantages
    if return_stats:
        return scores, scores, stats
    return scores, scores
