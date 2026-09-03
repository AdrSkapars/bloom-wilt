#!/usr/bin/env python3
"""Collect paired top-5 next-token distributions along a run's generated tokens.

Takes the tokens some run actually generated and teacher-forces that SAME sequence through
BOTH contexts, recording the top-5 alternatives the model would have considered at each
position:

  target    the conversation with the TARGET system prompt, as a vanilla run sees it
  elicited  the same conversation with the target system prompt REPLACED by the jail
            persona and the behaviour file's prefill appended -- the `b1=0, b2=1` context

The two contexts are always these; `--arm` chooses whose TOKENS they are forced along:

  --arm vanilla_15s       target-only tokens   (what the target itself said)
  --arm api_elicited_15s  elicited-only tokens (what the jail context said)

Both directions matter because the position set is not symmetric: each run's tokens are
on-policy for its own context and off-policy for the other, so "how often do the top-5
lists overlap" has a different answer along each trajectory.

Aligned position-for-position, this is the raw material a top-k approximation to
`z = b1*l_target + b2*l_jail` would have to work from. Fireworks caps `logprobs` at 5.

The sequence is sent as TOKEN IDS, not text -- sampling can produce a non-canonical
tokenization, and re-tokenizing the decoded string would silently score a different
sequence (see apitilt.py:score_ids). The echo is verified to come back unchanged.

`top_logprobs[i]` is the distribution that PREDICTED token i, so the sampled token is
frequently absent from its own top-5. That is signal, not an error.

Output: one JSON per transcript in <run>/top5/, plus top5/_index.json.

  python -X utf8 experiments/postpaper/api_tilt/collect_top5.py --arm vanilla_15s
  python -X utf8 experiments/postpaper/api_tilt/collect_top5.py --arm api_elicited_15s
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "bloom"))

import yaml  # noqa: E402

# API keys live in the gitignored .env.local; load before importing the client.
_envf = REPO / ".env.local"
if _envf.exists():
    for _l in _envf.read_text(encoding="utf-8").replace("\r", "").split("\n"):
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
os.environ.setdefault("BLOOM_TARGET_CHAT_TEMPLATE",
                      str(REPO / "src/bloom/prompts/deepseek_v4_chat.jinja"))
os.environ.setdefault("BLOOM_TARGET_API", "fireworks")

from bloom.apitilt import load_api_target  # noqa: E402

TARGET = "api/accounts/fireworks/models/deepseek-v4-flash-0731"
ARM_ROOT = REPO / "experiments/postpaper/runs_dsv4/self_harm/deepseek_v4_flash"
BEHAVIOUR = REPO / "src/bloom/prompts/7_self_harm_encouragement.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="vanilla_15s",
                    help="source run whose TOKENS are forced through both contexts "
                         "(vanilla_15s = target-only tokens; api_elicited_15s = elicited-only "
                         "tokens). The two contexts scored are the same either way.")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N transcripts")
    ap.add_argument("--top-k", type=int, default=5, help="alternatives per position (Fireworks max 5)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src_run = ARM_ROOT / args.arm / f"round_{args.round}"
    if not (src_run / "transcripts").is_dir():
        raise SystemExit(f"no transcripts under {src_run}")

    beh = yaml.safe_load(open(BEHAVIOUR, encoding="utf-8"))
    jail_sys = beh["jailbroken_output_system_prompt"]
    jail_prefill = beh.get("jailbroken_output_prefill", "") or ""

    handle = load_api_target(TARGET)
    client = handle["client"]
    # Registry-derived no-think wrappers; '' for DeepSeek-V4 (its template closes </think>).
    nt, nt_c = handle["target_no_think"], handle["corrupt_no_think"]

    out_dir = src_run / "top5"
    out_dir.mkdir(exist_ok=True)

    tfs = sorted((src_run / "transcripts").glob("transcript_v*r*.json"))
    if args.limit:
        tfs = tfs[:args.limit]
    print(f"{len(tfs)} transcripts from {src_run.relative_to(REPO)}")

    def build_jobs(tf):
        """One job per (transcript, assistant turn): the two prefixes + the forced ids."""
        d = json.load(open(tf, encoding="utf-8"))
        jobs, ctx = [], []
        for m in d["messages"]:
            gid = m.get("gen_token_ids")
            if m["role"] == "assistant" and gid:
                # target context: the conversation exactly as it ran
                t_prefix = client.render(ctx, add_generation_prompt=True) + nt
                # elicited context: drop the target system prompt, prepend the jail persona,
                # append the prefill (which conditions but is never sampled)
                conv = [x for x in ctx if x.get("role") != "system"]
                j_msgs = ([{"role": "system", "content": jail_sys}] + conv) if jail_sys else conv
                j_prefix = (client.render(j_msgs, add_generation_prompt=True) + nt_c + jail_prefill)
                jobs.append({"turn": len(jobs) + 1, "ids": list(gid),
                             "t_prefix": t_prefix, "j_prefix": j_prefix})
            ctx = ctx + [{"role": m["role"], "content": m["content"]}]
        return d, jobs

    def run_one(tf):
        d, jobs = build_jobs(tf)
        turns = []
        for j in jobs:
            tgt = client.score_ids_topk(j["t_prefix"], j["ids"], args.top_k)
            eli = client.score_ids_topk(j["j_prefix"], j["ids"], args.top_k)
            turns.append({
                "turn": j["turn"],
                "n_tokens": len(j["ids"]),
                "token_ids": j["ids"],
                "tokens": tgt["tokens"],
                # lp = logprob of the token that was actually generated, under that context.
                # top = [(token_string, logprob) x top_k], most likely first.
                "target":   {"lp": [round(v, 6) for v in tgt["lp"]],
                             "top": [[[t, round(v, 6)] for t, v in p] for p in tgt["top"]]},
                "elicited": {"lp": [round(v, 6) for v in eli["lp"]],
                             "top": [[[t, round(v, 6)] for t, v in p] for p in eli["top"]]},
            })
        rec = {
            "meta": {
                "transcript": tf.name,
                "variation_number": d["metadata"]["variation_number"],
                "repetition_number": d["metadata"]["repetition_number"],
                "source_run": str(src_run.relative_to(REPO)).replace("\\", "/"),
                "source_target_model": d["metadata"]["target_model"],
                "scored_by": TARGET,
                "top_k": args.top_k,
                "target_system_prompt": d["metadata"]["target_system_prompt"],
                "jail_system_prompt": jail_sys,
                "jail_prefill": jail_prefill,
                "forced_along": args.arm,   # whose tokens these are
                "note": (f"tokens are the {args.arm} run's generation; BOTH contexts are "
                         "teacher-forced along them. top[i] is the distribution that "
                         "predicted token i, so the sampled token need not appear in it."),
            },
            "turns": turns,
        }
        op = out_dir / tf.name.replace(".json", ".top5.json")
        op.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return tf.name, sum(t["n_tokens"] for t in turns), op

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name, ntok, op in ex.map(run_one, tfs):
            results.append({"transcript": name, "n_tokens": ntok,
                            "file": op.name, "bytes": op.stat().st_size})
            print(f"  {name}: {ntok} positions x2 contexts -> {op.name}", flush=True)

    index = {"source_run": str(src_run.relative_to(REPO)).replace("\\", "/"),
             "forced_along": args.arm,
             "scored_by": TARGET, "top_k": args.top_k,
             "n_transcripts": len(results),
             "n_positions": sum(r["n_tokens"] for r in results),
             "files": sorted(results, key=lambda r: r["transcript"])}
    (out_dir / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    mb = sum(r["bytes"] for r in results) / 1e6
    print(f"\n{index['n_positions']} positions across {len(results)} transcripts, "
          f"{mb:.1f} MB in {out_dir.relative_to(REPO)}")
    print(f"api: {client.stats()}   wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
