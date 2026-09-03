# Working in this repo

This repo has two jobs at once: it is the archive of the code the paper's numbers came
from, and it is where post-paper work happens. Keeping those from interfering is the
main convention here.

## The paper's code is tagged

`git tag paper` -> `d9f5f4d` ("Add the arXiv citation", 2026-09-01), the last commit
before any DeepSeek-V4, sharding or hosted-API work. Everything the paper's tables and
figures were produced from is at or before that point. `git checkout paper` recovers it.

Post-paper work continues on `main`. Do not branch for it — a long-lived branch splits
where the runs live and buys nothing the tag does not.

## Where new code goes

| | |
|---|---|
| `experiments/bloom/` | The paper's tree — runs, helpers, banks. Frozen. Read it, don't edit it. The kickoff banks under `_banks/` are reused by new runs on purpose, so every arm sees identical scenarios. |
| `experiments/postpaper/` | Everything since. Runners, analysis scripts, new runs. **New code goes here by default.** |
| `src/bloom/` | The shared pipeline. Only touch it when the work genuinely cannot live in a runner — a new decode engine, for instance, has to be dispatched from inside the rollout loop. |

## Editing the shared pipeline

When `src/` does have to change, every edit to a file the paper's runs execute
(`core.py`, `wilt.py`, `rollout.py`, `pipeline.py`, `search.py`, `bloom_corrupt.py`)
must be **additive and gated**: guarded by a condition that is false for every paper
config, so the paper's code path is bit-identical.

The `api_tilt` engine is the worked example. Its hooks are all behind
`target_is_api` (`BLOOM_TARGET_MODEL` starting `api/`) or
`jail_runtime_cfg["engine"] == "api_tilt"`; the engine itself is a self-contained
module (`src/bloom/bloom/apitilt.py`) imported lazily at the two call sites. Nothing
on the local path changed behaviour.

Prefer, in order: put it in a runner under `experiments/postpaper/`; add a
self-contained module in `src/bloom/bloom/` plus a gated dispatch; edit shared logic
in place. The last one needs a reason.

After changing a shared file, check the diff and confirm every hunk is either inside a
new module, inside a gate, or provably inert for a local target:

```bash
git diff paper -- src/bloom/bloom/rollout.py
```

## Two operational gotchas

- **`PYTHONIOENCODING=utf-8` is required** on Windows for anything that prints a
  rendered prompt — the DeepSeek-V4 chat template contains U+FF5C and cp1252 stdout
  raises on it.
- `uv sync` does not work on the Windows laptop (`pyproject.toml` pins vllm). The
  API-only path needs just `litellm`, `tenacity`, `pyyaml`, `jinja2`.
