# postpaper

Everything built after the paper. `experiments/bloom/` is frozen as the paper's tree —
its runs, helpers and banks are what the tables and figures were produced from — so new
code and new runs land here instead.

| | |
|---|---|
| `deepseek_v4/` | Scaling WILT to a near-frontier target (DeepSeek-V4-Flash, 284B/13B-active). Setup runbook, launcher, selection plots. |
| `runs_dsv4/`   | All DeepSeek-V4 runs, local and hosted. Layout is unchanged from `experiments/bloom/runs_new`: `<behaviour>/<model>/<arm>/round_N/`. |
| `api_tilt/`    | Running the target over a hosted API instead of local weights — which parts of LogitTilt survive without logits, and the top-k approximation to the parts that don't. |

The paper's code is tagged `paper` (`git checkout paper`). Edits to the shared pipeline
in `src/` since that tag are additive and gated so the paper's path is unchanged — see
`CLAUDE.md` for the rule.

Runs still address as `BLOOM_FOLDER=runs_dsv4/...`; the launchers set
`BLOOM_RUNS_ROOT=experiments/postpaper` so that resolves here rather than into the paper's
tree. The kickoff bank they reuse still lives at
`experiments/bloom/_banks/runs_hyperparam/self_harm/Qwen_Qwen3.5-4B/_bank` — it is paper
data being reused deliberately, so every arm sees identical scenarios and kickoffs.
