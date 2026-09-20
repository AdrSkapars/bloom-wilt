#!/usr/bin/env python3
"""How many distinct outputs carry the probability mass, for a fixed input and length?

The output tree has V**L leaves -- 248320**20 is about 1e107 for a 20-token reply -- so any
fraction of that is meaningless. What carries information is:

  N_eff = exp(H)   the effective number of outputs, from the sequence entropy H
  survival(k)      the probability that a sample stays inside the per-position top-k the
                   whole way, i.e. the mass that survives truncating every branch outside
                   the top k
  mass_above(X)    the mass carried by sequences whose OWN probability is at least X, i.e.
                   what survives pruning on the full-output probability

The last two are the two pruning rules, and both are plain fractions of sampled paths:
sampling from P makes the fraction of samples in a set an unbiased estimate of that set's
mass. So one sampling pass answers both.

What sampling CANNOT give is N_p, the minimum count of sequences covering mass p. With a
large support the same sequence is essentially never drawn twice, so there are no repeat
statistics to extrapolate from -- that needs a best-first enumeration, which is worth
building only once these numbers say it is tractable.

Two estimators of H are computed and should agree, which is the run's own sanity check:

  H_mc     mean of -log P(y) over samples. Unbiased, high variance.
  H_chain  mean of the summed per-position conditional entropies. Also unbiased, much lower
           variance, because it uses the whole distribution at each step rather than only
           the token that happened to be drawn.

Token-level counting deliberately conflates paraphrase with genuine alternatives: "Also"
versus "Additionally" are two leaves here. That is the question being asked, not an
oversight.

    python branching.py run --beh racial --model qwen --scen 0 --len 20 --n 512
    python branching.py report
"""
import argparse
import glob
import io
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(REPO, "experiments", "postpaper", "oracle_check"))
sys.path.insert(0, os.path.join(REPO, "src"))

from oracle_check import MODELS, THINK_PREFILL, _prompts, _scenarios  # noqa: E402

KS = [1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 5000, 50000]


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bloom.bloom import core

    mid = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    pre = THINK_PREFILL if core.uses_think_block(mid) else ""
    t_sys, e_sys, e_pre, _, _ = _prompts(a.beh)
    scen = _scenarios(a.beh)[a.scen]

    if a.ctx == "elicited":
        msgs = ([{"role": "system", "content": e_sys}] if e_sys else []) + \
               [{"role": "user", "content": scen}]
        ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + pre + e_pre
    else:
        msgs = [{"role": "system", "content": t_sys}, {"role": "user", "content": scen}]
        ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + pre

    ids = tok.encode(ctx, add_special_tokens=False)
    V = model.config.vocab_size

    def _batch(B):
        """Sample B sequences of exactly `a.len` tokens, returning per-sequence records.

        Run in chunks rather than all at once: the KV cache for hundreds of sequences over a
        ~500-token prompt, plus several [B, V] float32 tensors at a 248k vocab (half a
        gigabyte each at B=512), overruns a 48GB card. Chunking costs nothing -- the samples
        are independent -- and keeps the peak flat in the number requested.
        """
        import torch
        inp = torch.tensor([ids] * B, device="cuda:0")
        past = None
        # per step, per sequence: logprob of the drawn token, entropy of the full conditional,
        # and the RANK of the drawn token. The rank is what makes every top-k answerable from
        # one pass: a path survives top-k truncation exactly when its worst rank is below k.
        lps, ents, ranks = [], [], []
        with torch.no_grad():
            for t in range(a.len):
                out = model(input_ids=inp if past is None else inp[:, -1:],
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                lg = out.logits[:, -1, :].float()
                if tok.eos_token_id is not None:
                    lg[:, tok.eos_token_id] = -float("inf")   # fixed length: EOS must not end
                lp = torch.log_softmax(lg, dim=-1)
                p = lp.exp()
                # EOS was masked to -inf, so that entry has p=0 and lp=-inf and the
                # product is 0*-inf = nan, which poisons the whole sum. Zero the log
                # where the probability is zero, which is the limit p*log p -> 0.
                lpz = torch.where(p > 0, lp, torch.zeros_like(lp))
                ents.append((-(p * lpz).sum(-1)).tolist())
                nxt = torch.multinomial(p, 1)
                got = lp.gather(-1, nxt)
                lps.append(got.squeeze(-1).tolist())
                # rank = how many tokens are strictly more likely than the drawn one
                ranks.append((lp > got).sum(-1).tolist())
                inp = torch.cat([inp, nxt], dim=-1)
                del out, lg, lp, lpz, p, got
        rec = []
        for i in range(B):
            rec.append({"logp": sum(lps[t][i] for t in range(a.len)),
                        "sum_ent": sum(ents[t][i] for t in range(a.len)),
                        "max_rank": max(ranks[t][i] for t in range(a.len)),
                        "text": tok.decode(inp[i][len(ids):], skip_special_tokens=True)})
        del inp, past
        torch.cuda.empty_cache()
        return rec

    seqs = []
    while len(seqs) < a.n:
        b = min(a.chunk, a.n - len(seqs))
        seqs += _batch(b)
        print("  sampled %d/%d" % (len(seqs), a.n), flush=True)
    B = len(seqs)
    os.makedirs(OUT, exist_ok=True)
    rec = {"model": a.model, "beh": a.beh, "scen": a.scen, "ctx": a.ctx, "len": a.len,
           "n": B, "vocab": V, "seqs": seqs}
    p = os.path.join(OUT, "br_%s_%s_s%d_%s_L%d_n%d.json"
                     % (a.beh, a.model, a.scen, a.ctx, a.len, B))
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        json.dump(rec, f)
    print("wrote %s" % os.path.basename(p))


def cmd_report(a):
    for p in sorted(glob.glob(os.path.join(OUT, "br_*.json"))):
        r = json.load(io.open(p, "r", encoding="utf-8"))
        s = r["seqs"]
        L, V, n = r["len"], r["vocab"], r["n"]
        h_mc = -st.mean(x["logp"] for x in s)
        h_chain = st.mean(x["sum_ent"] for x in s)
        print("\n=== %s ===" % os.path.basename(p)[3:-5])
        print("  L=%d  n=%d  vocab=%d   theoretical leaves V**L = 10^%.0f"
              % (L, n, V, L * math.log10(V)))
        print("  H (mean -logP)      = %8.2f nats   -> N_eff = 10^%.1f" % (h_mc, h_mc / math.log(10)))
        print("  H (summed entropy)  = %8.2f nats   -> N_eff = 10^%.1f"
              % (h_chain, h_chain / math.log(10)))
        print("  bits/token          = %8.2f" % (h_chain / L / math.log(2)))
        print("  spread of -logP     : sd %.2f nats, min %.2f, max %.2f"
              % (st.pstdev([-x["logp"] for x in s]), -max(x["logp"] for x in s),
                 -min(x["logp"] for x in s)))
        print("\n  survival(k): mass that never leaves the per-position top-k")
        for k in KS:
            if k >= V:
                continue
            f = sum(1 for x in s if x["max_rank"] < k) / n
            print("    k=%-6d %6.1f%%   %s" % (k, 100 * f, "#" * int(round(40 * f))))
        print("\n  mass_above(X): mass on sequences whose own probability is >= X")
        for e in (-4, -6, -8, -10, -14, -20, -30):
            f = sum(1 for x in s if x["logp"] >= e * math.log(10)) / n
            print("    P>=1e%-4d %6.1f%%" % (e, 100 * f))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.set_defaults(fn=cmd_run)
    p.add_argument("--beh", default="racial")
    p.add_argument("--model", default="qwen", choices=sorted(MODELS))
    p.add_argument("--scen", type=int, default=0)
    p.add_argument("--ctx", default="target", choices=["target", "elicited"])
    p.add_argument("--len", type=int, default=20)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--chunk", type=int, default=32,
                   help="sequences sampled per forward batch; the total --n is reached by "
                        "repeating, which keeps peak memory flat in --n")
    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)
