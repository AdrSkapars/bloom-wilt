# BLOOM-WILT: Logit Tilting Behaviour Elicitation for Automated LLM Auditing

This is the research code behind the paper. To use LogitTilt in your own evaluations,
install the [Inspect](https://inspect.aisi.org.uk) extension instead — it is the steering
on its own, without the rest of the pipeline:

```bash
pip install inspect-logittilt
```

**[Documentation](https://adrskapars.github.io/inspect_logittilt/)** ·
**[Repository](https://github.com/AdrSkapars/inspect_logittilt)** ·
**[PyPI](https://pypi.org/project/inspect-logittilt/)**

Every audit transcript, judge score, and run configuration behind the paper is published
separately as a Hugging Face dataset:
**[AdrSkapars/bloom-wilt-transcripts](https://huggingface.co/datasets/AdrSkapars/bloom-wilt-transcripts)**

## Requirements

```bash
uv sync
```

An audit runs two models: an **auditor** (writes the adversarial conversation, judges the
result) and a **target** (the model under audit). We run these on 2×46 GB GPUs, one each.
They fit on a single GPU if it is large enough — set `BLOOM_EVAL_GPU` and `BLOOM_TARGET_GPU`
to the same id and lower the two `gpu_memory_utilization` values. Alternatively, use a hosted
auditor and keep only the target local:

```bash
BLOOM_EVAL_MODEL=claude-sonnet-4-6 BLOOM_EVAL_THINKING=0 python src/bloom/bloom_corrupt.py
```

`transformers>=5.13` and `vllm>=0.24` are pinned deliberately: earlier versions lack the
Gemma-4 architecture, and an unpinned resolve silently falls back to them.

## Running an audit

There are no command-line arguments. Everything is configured through the `CFG` dict at the
top of [`src/bloom/bloom_corrupt.py`](src/bloom/bloom_corrupt.py), optionally overridden by
environment variables. Defaults give you a plain best-of-N baseline: Qwen3.5-4B as target,
Gemma-4-26B-A4B as auditor, racial bias, 5 rounds × 3 turns.

```bash
python src/bloom/bloom_corrupt.py
```

**LogitTilt** — set `BLOOM_JAIL_MODEL` to switch on output-side steering. `self` uses the
target as its own steering expert, which is what the paper does:

```bash
BLOOM_JAIL_MODEL=self BLOOM_JAIL_BETA=1.5 \
BLOOM_BEHAVIOR_FILE=prompts/7_self_harm_encouragement.yaml \
BLOOM_FOLDER=runs/my_logittilt \
python src/bloom/bloom_corrupt.py
```

**G-PAIR** — set `BLOOM_REFINE=1` to switch on input-side refinement, where each round's
opening message is rewritten in light of previous rounds:

```bash
BLOOM_REFINE=1 \
BLOOM_BEHAVIOR_FILE=prompts/7_self_harm_encouragement.yaml \
BLOOM_FOLDER=runs/my_gpair \
python src/bloom/bloom_corrupt.py
```

**WILT** is both at once — set `BLOOM_JAIL_MODEL` and `BLOOM_REFINE` together.

##

To read transcripts in a browser rather than as raw JSON, launch the viewer on a results
folder (needs `npm`; the path is relative to `experiments/bloom/`, and omitting it picks the
most recent `runs_*`):

```bash
python experiments/bloom/helpers/viewer.py runs/my_logittilt
```


## Configuration

The `CFG` dict is the source of truth and every key is commented in place; the environment
variables below override it. There are many more `BLOOM_*` knobs in the code for ablations
and for the other methods in the paper — these are the ones needed for LogitTilt and G-PAIR.

### Core

| Variable | Default | Meaning |
|---|---|---|
| `BLOOM_TARGET_MODEL` | `local/Qwen/Qwen3.5-4B` | Model under audit. Must be `local/<hf-repo>`. Paper targets: `local/meta-llama/Llama-3.2-3B-Instruct`, `local/microsoft/Phi-4-mini-instruct`, `local/Qwen/Qwen3.5-4B`, `local/google/gemma-4-e4b-it` |
| `BLOOM_EVAL_MODEL` | `local/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` | Auditor. A non-`local/` id (e.g. `claude-sonnet-4-6`) routes to a hosted API via litellm |
| `BLOOM_JUDGE_MODEL` | same as auditor | Scores behaviour presence. Also accepts a hosted id |
| `BLOOM_BEHAVIOR_FILE` | `prompts/1_racial_bias.yaml` | Behaviour spec, relative to `src/bloom/`. One of `1_racial_bias`, `2_political_bias`, `3_reinforcing_delusions`, `4_strategic_deception`, `5_self_preservation`, `6_reward_hacking`, `7_self_harm_encouragement`, `8_dangerous_medical_advice`, `9_goblin_fixation` |
| `BLOOM_FOLDER` | `runs/default` | Output directory, under `experiments/bloom/` |
| `BLOOM_NUM_SCENARIOS` | `100` | Scenarios the auditor invents |
| `BLOOM_NUM_ROUNDS` | `5` | Rounds; round 1 runs the full pipeline, rounds 2+ re-roll and re-judge |
| `BLOOM_MAX_TURNS` | `3` | Conversation turns per rollout |
| `BLOOM_SEED` | `100` | Base seed; round *R* uses `seed + R` |
| `BLOOM_EVAL_GPU` / `BLOOM_TARGET_GPU` | `0` / `1` | Device ids |
| `BLOOM_KICKOFF_BANK` | unset | Reuse another run's scenarios so two methods see identical inputs |

### LogitTilt (`jailbroken_output`)

| Variable | Default | Meaning |
|---|---|---|
| `BLOOM_JAIL_MODEL` | unset (off) | Setting it enables LogitTilt. `self` = target steers itself; otherwise a `local/` model sharing the target's vocabulary |
| `BLOOM_JAIL_BETA` | `4.0` | Steering strength β. Tuned per model and behaviour; the paper sweeps 0.5–4.0 in steps of 0.5 |
| `BLOOM_JAIL_PREFILL` | `1` | Compliance prefill on the steering expert |
| `BLOOM_JAIL_FLOOR` | unset | Restrict steering to tokens the unmodified target already gives at least this probability |

### G-PAIR (`refinement_input`)

| Variable | Default | Meaning |
|---|---|---|
| `BLOOM_REFINE` | `0` | `1` enables refinement. Off, each round is an independent resample |
| `BLOOM_REFINE_HIST_TRANSCRIPT` | `2` | Prior full transcripts shown when refining: `all`, `0`, or *N* |
| `BLOOM_REFINE_HIST_STRATEGY` | `all` | Prior (round, score, strategy) rows shown |

## Output

Each run writes one directory per round:

```
experiments/bloom/<BLOOM_FOLDER>/round_1/
├── cfg.json            # the fully resolved configuration
├── understanding.json  # auditor's reading of the behaviour spec
├── ideation.json       # the scenarios it invented
├── judgment.json       # per-scenario scores + summary_statistics
└── transcripts/        # one JSON per scenario
```

`judgment.json` → `summary_statistics` carries `average_behavior_presence_score` (1–10; the
paper reports ×10) and `elicitation_rate`. Per-transcript token probabilities are in each
transcript's `prob_stats`.

## Layout

```
src/bloom/bloom_corrupt.py   entry point and CFG
src/bloom/bloom/             core, pipeline, rollout, wilt, search, flrt
src/bloom/prompts/           behaviour specs + shared prompt templates
experiments/bloom/helpers/   analysis and plotting for the paper
paper/                       ICML and NeurIPS sources
```

The scripts in `helpers/` read a runs tree that is no longer in this repository — fetch it
from the Hugging Face dataset above first.
