# SGPO: State-Graph Policy Optimization

SGPO (State-Graph Policy Optimization) is a reinforcement-learning advantage estimator for
**multi-turn LLM agent tasks**, built on the [verl-agent](https://github.com/langfengQ/verl-agent) /
verl framework with [GiGPO](https://arxiv.org/abs/2505.10978) as its baseline.

## Layout

```
recipe/sgpo/          # SGPO implementation
  core_sgpo.py        # the SGPO advantage estimator
  rewind.py           # optional counterfactual rewind branching
  test_*.py
gigpo/core_gigpo.py   # GiGPO baseline, reused by SGPO
verl/                 # verl training framework
  trainer/ppo/ray_trainer.py        # advantage-estimator dispatch
  trainer/config/ppo_trainer.yaml   # gigpo / sgpo config blocks
agent_system/         # agent rollout loop + WebShop/ALFWorld env managers
examples/sgpo_trainer/{run_webshop_sgpo,run_alfworld_sgpo}.sh
```

## Usage

Pick the estimator via `algorithm.adv_estimator` ∈ `{gigpo, sgpo}`. Example launchers live in
`examples/sgpo_trainer/`; point them at your own model (`MODEL_PATH`) and parquet goals (`DATA_DIR`).

```bash
bash examples/sgpo_trainer/run_webshop_sgpo.sh
bash examples/sgpo_trainer/run_alfworld_sgpo.sh
```

Main `sgpo` knobs (`verl/trainer/config/ppo_trainer.yaml`):

| Key | Meaning |
| --- | --- |
| `algorithm.sgpo.dp_w` | step-term weight; `0` reduces to episode-level GRPO |
| `algorithm.sgpo.lam` | TD(λ); `null` = cross-validated, a float pins it (`1.0` = GiGPO) |

## Tests

```bash
python -m recipe.sgpo.test_sgpo
python -m recipe.sgpo.test_rewind
```

