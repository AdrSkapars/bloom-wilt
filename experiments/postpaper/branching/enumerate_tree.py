#!/usr/bin/env python3
"""Count the replies exactly, instead of estimating how many there are.

The sampling script gives an effective count via the entropy: how many equally likely
replies the model's behaviour is *equivalent* to. It cannot give the actual number needed to
cover a given share of the probability, because with a large support the same reply is
essentially never drawn twice, so there are no repeats to extrapolate from.

At short lengths the tree is small enough to walk outright. This expands every branch whose
running probability stays above a floor, which is the second pruning rule from the original
question -- "going down this path would put the full output probability under X, so prune
it". What comes back is the complete list of surviving replies with their exact
probabilities, so the answer to "how many replies carry 90% of the mass" is a count, not an
estimate.

The floor is what keeps this finite, and it is also the one thing that can invalidate the
answer: mass that falls below the floor is mass we never see. So the total captured
probability is reported alongside every count. If capture is 0.97 then a "99% of mass"
question cannot be answered at that floor, and the floor has to come down.

    python enumerate_tree.py --beh racial --scen 0 --len 5 --floor 1e-6
"""
import argparse
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(REPO, "experiments", "postpaper", "oracle_check"))
sys.path.insert(0, os.path.join(REPO, "src"))

from oracle_check import MODELS, THINK_PREFILL, _prompts, _scenarios  # noqa: E402


def main(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bloom.bloom import core

    mid = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    pre = THINK_PREFILL if core.uses_think_block(mid) else ""
    t_sys, _, _, _, _ = _prompts(a.beh)
    scen = a.prompt if a.prompt else _scenarios(a.beh)[a.scen]
    ctx = tok.apply_chat_template(
        [{"role": "system", "content": t_sys}, {"role": "user", "content": scen}],
        tokenize=False, add_generation_prompt=True) + pre
    ids = tok.encode(ctx, add_special_tokens=False)

    logfloor = math.log(a.floor)
    # frontier: (token ids so far, log probability so far). Grown one level at a time so the
    # whole level can go through the model as one batch.
    frontier = [([], 0.0)]
    done = []            # paths that ended by sampling EOS, at whatever depth
    pruned_mass = 0.0
    pruned_count = 0

    # LOOKAHEAD PRUNING. A branch is worth keeping only if it could still clear the floor once
    # finished. The best conceivable finish is every remaining token at probability 1, which
    # adds 0 to the log-total -- so with mstar=0 the test is exactly "is the running total
    # already under the floor", and it is admissible: nothing that could have qualified is
    # ever discarded, because probability only falls as a path grows.
    #
    # mstar < 0 replaces "perfect" with "realistically good": a typical strong token scores
    # about -0.05 to -0.2 in log terms, so assuming that instead of 0 tightens the bound and
    # prunes far more. That is no longer a guarantee -- a path whose remaining tokens really
    # are near-certain can now be cut by mistake -- which is why the exact setting is the
    # default and the two are meant to be run side by side and compared.
    #
    # Pruning on the TOTAL rather than a per-token mean is what keeps this valid when EOS can
    # end a path early: totals only decrease, whatever depth a path stops at, so paths of
    # different lengths need no normalisation to be compared against one floor.
    def _keeps(logp, depth):
        return logp + (a.len - depth) * a.mstar >= logfloor
    for depth in range(a.len):
        if not frontier:
            break
        nxt = []
        for i in range(0, len(frontier), a.batch):
            chunk = frontier[i:i + a.batch]
            inp = torch.tensor([ids + c[0] for c in chunk], device="cuda:0")
            with torch.no_grad():
                lg = model(input_ids=inp).logits[:, -1, :].float()
            if tok.eos_token_id is not None and not a.allow_eos:
                lg[:, tok.eos_token_id] = -float("inf")
            lp = torch.log_softmax(lg, dim=-1)
            # only the top `width` children can matter: anything below the width-th child is
            # smaller still, so if that one is under the floor the rest are too.
            top = lp.topk(a.width, dim=-1)
            for j, (seq, slp) in enumerate(chunk):
                for r in range(a.width):
                    tokid = int(top.indices[j, r])
                    child = slp + float(top.values[j, r])
                    if not _keeps(child, depth + 1):
                        pruned_mass += math.exp(child)
                        pruned_count += 1
                        continue
                    if a.allow_eos and tok.eos_token_id is not None and tokid == tok.eos_token_id:
                        done.append((seq, child))     # finished here, do not extend
                        continue
                    nxt.append((seq + [tokid], child))
        frontier = nxt
        print("  depth %d: %d live branches" % (depth + 1, len(frontier)), flush=True)
        if len(frontier) > a.max_nodes:
            print("  STOPPED: over --max-nodes, the floor is too low for this length")
            break

    # whatever is still alive at max depth counts as a reply too, alongside the ones that
    # ended on EOS
    frontier = done + frontier
    frontier.sort(key=lambda x: -x[1])
    total = sum(math.exp(p) for _, p in frontier)
    print("\n%d complete replies above the floor" % len(frontier))
    if a.allow_eos:
        print("   of which %d ended on EOS, %d ran to the full %d tokens"
              % (len(done), len(frontier) - len(done), a.len))
    print("captured probability: %.4f   (pruned away: %.4f over %d branches)"
          % (total, pruned_mass, pruned_count))
    print("bound: mstar=%g  (%s)"
          % (a.mstar, "exact, nothing lost" if a.mstar == 0 else "optimistic, may lose real paths"))
    print("\nreplies needed to cover a share of the CAPTURED mass:")
    run = 0.0
    marks = [0.5, 0.8, 0.9, 0.95, 0.99]
    mi = 0
    for n, (_, p) in enumerate(frontier, 1):
        run += math.exp(p)
        while mi < len(marks) and run >= marks[mi] * total:
            print("   %4.0f%% of mass  <- %d replies (%.2f%% of the %d found)"
                  % (100 * marks[mi], n, 100.0 * n / len(frontier), len(frontier)))
            mi += 1
    print("\ntop 5 replies:")
    for _, p in frontier[:5]:
        pass
    for seq, p in frontier[:5]:
        print("   p=%.4g  %r" % (math.exp(p), tok.decode(seq, skip_special_tokens=True)))
    os.makedirs(OUT, exist_ok=True)
    tag = a.tag or ("%s_s%d" % (a.beh, a.scen))
    fp = os.path.join(OUT, "enum_%s_%s_L%d_f%g_m%g%s.json"
                      % (tag, a.model, a.len, a.floor, a.mstar, "_eos" if a.allow_eos else ""))
    with io.open(fp, "w", encoding="utf-8", newline="") as f:
        json.dump({"tag": tag, "model": a.model, "len": a.len, "floor": a.floor,
                   "captured": total, "pruned": pruned_mass,
                   "mstar": a.mstar, "allow_eos": bool(a.allow_eos),
                   "pruned_count": pruned_count,
                   "replies": [{"logp": p, "ntok": len(s),
                                "mean_logp": (p / len(s)) if s else 0.0,
                                "text": tok.decode(s, skip_special_tokens=True)}
                               for s, p in frontier]}, f)
    print("\nwrote %s" % os.path.basename(fp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beh", default="racial")
    ap.add_argument("--scen", type=int, default=0)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default="qwen", choices=sorted(MODELS))
    ap.add_argument("--len", type=int, default=5)
    ap.add_argument("--floor", type=float, default=1e-6,
                    help="prune any branch whose running probability drops below this")
    ap.add_argument("--width", type=int, default=64,
                    help="children considered per node; beyond the top few dozen everything "
                         "is far under any sane floor")
    ap.add_argument("--allow-eos", action="store_true",
                    help="let EOS be sampled as a normal token, ending that path there")
    ap.add_argument("--mstar", type=float, default=0.0,
                    help="assumed per-token log-prob for the remaining tokens when deciding "
                         "whether a branch can still clear the floor. 0 is the exact bound "
                         "(prunes nothing that could qualify); negative values are tighter "
                         "and faster but can discard real paths")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-nodes", type=int, default=200000)
    main(ap.parse_args())
