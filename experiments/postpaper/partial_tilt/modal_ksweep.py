"""Run the partial_tilt top-k sweep on Modal. UNTESTED -- see "First run" below.

Why this is a small job, despite the paper's usual GPU footprint:

  * self-jail shares weights. wilt._load_hf_poe_models sets mc = mt when the elicited model
    IS the target, so Qwen3.5-4B costs ~8GB once, not twice. Both contexts are separate KV
    caches over one set of weights.
  * vLLM is never imported. core imports it inside _vllm_worker_main, which only runs for a
    `local/` auditor. With AUDITOR=api the auditor is OpenRouter, no worker spawns, and the
    image skips the heaviest and most version-fragile dependency in the project.

So one 24GB card runs the whole sweep, and the sweep is a handful of dollars.

Note the auditor is then OpenRouter's Gemma-4-26B rather than the paper's local FP8 copy.
That is fine HERE because the auditor is held fixed across every k -- it cannot confound a
comparison between k values -- and it is the same auditor the hosted api_tilt runs used. It
would not be fine for a number meant to sit in a table beside the paper's.

Layout:
  /cache/hf   Volume, HF weights. Qwen downloads once and every later run is an SSD read.
  /runs       Volume, output. Committed after each k so a crash keeps what finished.

First run, from the repo root:

    pip install modal && modal setup
    modal secret create bloom-keys OPENROUTER_API_KEY=sk-...    # never checked in
    modal run experiments/postpaper/partial_tilt/modal_ksweep.py --ks 0

Start with k=0 ALONE. It is the anchor: no truncation, so the engine must reproduce the
paper's full-vocab LogitTilt exactly. Check it against a jailbroken_output run at the same b2
before spending anything on k>0, because every truncated point is measured against it.

Then the sweep, and pull the results down:

    modal run experiments/postpaper/partial_tilt/modal_ksweep.py --ks 1,2,3,5,10,20,50
    modal volume get bloom-runs /runs_local ./experiments/postpaper/runs_local
    python -X utf8 experiments/postpaper/partial_tilt/ksweep.py self_harm Qwen_Qwen3.5-4B
"""
import os
import subprocess
import sys

import modal

REPO = "/root/bloom-wilt"
HF_CACHE = "/cache/hf"
RUNS = "/runs"

# No vllm: see the module docstring. torch is the cu124 wheel Modal's CUDA image expects.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "accelerate>=1.0",
        "litellm>=1.60",
        "tenacity>=8.2",
        "pyyaml>=6.0",
        "jinja2>=3.1",
        "huggingface_hub>=0.30",
    )
    .env({
        "HF_HOME": HF_CACHE,
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    })
    # The repo itself. add_local_dir is applied last so a code edit does not invalidate the
    # (slow) pip layer above -- iterating on hftilt.py should not mean reinstalling torch.
    .add_local_dir(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        remote_path=REPO,
        ignore=["~*", ".git", "**/__pycache__", "**/runs_*", "paper", "*.pdf"],
    )
)

app = modal.App("bloom-partial-tilt")
hf_vol = modal.Volume.from_name("bloom-hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("bloom-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10",                 # 24GB: ~8GB weights + two KV caches at var_batch=15.
                               # L4 is cheaper per hour but ~half the memory bandwidth, and a
                               # two-forward-passes-per-token decode is bandwidth-bound, so it
                               # can cost MORE per run. L40S if var_batch needs raising.
    volumes={HF_CACHE: hf_vol, RUNS: runs_vol},
    secrets=[modal.Secret.from_name("bloom-keys")],
    timeout=24 * 60 * 60,      # container ceiling; one k is minutes, the sweep is under an hour
)
def sweep(ks: str = "0", beh: str = "self_harm", model: str = "qwen",
          rule: str = "poe", b2: str = "1.5", scen: str = "15", var_batch: str = "15"):
    """One container, every k in turn. Each k is its own `bloom_corrupt.py` process, so the
    weights reload between them -- ~20s off the volume, which is not worth restructuring the
    pipeline to avoid."""
    os.chdir(REPO)

    # run_local.sh requires .env.local and sources it for keys. Rebuild it from the Modal
    # secret rather than shipping one: the file holds credentials and must never be in the
    # image or the repo.
    with open(os.path.join(REPO, ".env.local"), "w", encoding="utf-8") as fh:
        for key in ("OPENROUTER_API_KEY", "FIREWORKS_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"):
            if os.environ.get(key):
                fh.write("%s=%s\n" % (key, os.environ[key]))

    env = dict(os.environ)
    env.update({
        "BEH": beh, "MODEL": model, "RULE": rule, "B2": b2,
        "SCEN": scen, "VAR_BATCH": var_batch,
        # runs_local lives on the Volume; BLOOM_RUNS_ROOT is set by run_local.sh, so symlink
        # the tree it writes into the mount instead of forking the runner for Modal.
        "AUDITOR": "api",
    })

    local_runs = os.path.join(REPO, "experiments", "postpaper", "runs_local")
    os.makedirs(os.path.join(RUNS, "runs_local"), exist_ok=True)
    if not os.path.islink(local_runs):
        if os.path.isdir(local_runs):
            import shutil
            shutil.rmtree(local_runs)
        os.symlink(os.path.join(RUNS, "runs_local"), local_runs)

    failed = []
    for k in [x.strip() for x in ks.split(",") if x.strip()]:
        print("=" * 60, flush=True)
        print("### K=%s beh=%s model=%s rule=%s b2=%s" % (k, beh, model, rule, b2), flush=True)
        print("=" * 60, flush=True)
        rc = subprocess.call(
            ["bash", "experiments/postpaper/partial_tilt/run_local.sh", k],
            cwd=REPO, env=env)
        print("### K=%s exit=%d" % (k, rc), flush=True)
        if rc != 0:
            failed.append(k)
        # Commit per k: a later k crashing or the container dying must not cost the ones
        # that already finished.
        runs_vol.commit()
        hf_vol.commit()

    if failed:
        print("### FAILED k values: %s" % ",".join(failed), flush=True)
    return {"failed": failed}


@app.local_entrypoint()
def main(ks: str = "0", beh: str = "self_harm", model: str = "qwen",
         rule: str = "poe", b2: str = "1.5", scen: str = "15", var_batch: str = "15"):
    out = sweep.remote(ks=ks, beh=beh, model=model, rule=rule, b2=b2,
                       scen=scen, var_batch=var_batch)
    if out.get("failed"):
        print("FAILED: %s" % ",".join(out["failed"]))
        sys.exit(1)
    print("done -- pull results with:")
    print("  modal volume get bloom-runs /runs_local ./experiments/postpaper/runs_local")
