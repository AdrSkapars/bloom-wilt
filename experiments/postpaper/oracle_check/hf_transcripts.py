#!/usr/bin/env python3
"""Score REAL pipeline transcripts, instead of replies generated for this experiment.

The earlier oracle_check run paired a reply sampled from the elicited context against one
sampled from the target context, and asked which the contrast preferred. That test was
confounded: for self_harm and deception the judge scored the elicited-generated reply higher
in EVERY pair, so "came from the elicited prompt" and "is more behavioural" had identical
labels and the 93-100% scores could not tell the two apart.

These transcripts fix it. They come from a single unsteered arm, so every reply has the same
provenance, and the judge scores spread naturally across the scenarios -- racial/bon runs
from 0 to 100. Positives and negatives therefore differ in BEHAVIOUR and nothing else, which
is the comparison the oracle claim actually needs.

    # once, to fetch (see --help for the arm listing)
    python hf_transcripts.py fetch --beh racial --model qwen --arm bon
    python hf_transcripts.py score --beh racial --model qwen --arm bon
    python hf_transcripts.py report
"""
import argparse
import glob
import io
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "out")
CACHE = os.path.join(os.path.expanduser("~"), "hf_tx")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from oracle_check import ANTI, MODELS, THINK_PREFILL, _gather_lp, _logit_window, _prompts  # noqa: E402

DATASET = "AdrSkapars/bloom-wilt-transcripts"
MODELDIR = {
    "qwen": "Qwen_Qwen3.5-4B",
    "llama": "meta-llama_Llama-3.2-3B-Instruct",
    "phi": "microsoft_Phi-4-mini-instruct",
    "gemma": "google_gemma-4-e4b-it",
}


def cmd_fetch(a):
    from huggingface_hub import snapshot_download
    pat = "runs_final/%s/%s/%s/**" % (a.beh, MODELDIR[a.model], a.arm)
    snapshot_download(DATASET, repo_type="dataset", allow_patterns=[pat], local_dir=CACHE)
    n = len(glob.glob(os.path.join(CACHE, "runs_final", a.beh, MODELDIR[a.model], a.arm,
                                   "round_*", "transcripts", "*.json")))
    print("fetched %s/%s/%s -- %d transcripts" % (a.beh, a.model, a.arm, n))


def _transcripts(beh, model, arm):
    return sorted(glob.glob(os.path.join(CACHE, "runs_final", beh, MODELDIR[model], arm,
                                         "round_*", "transcripts", "*.json")))


def cmd_score(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bloom.bloom import core

    mid = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    think = core.uses_think_block(mid)
    pre = THINK_PREFILL if think else ""
    _, e_sys, e_pre, _, _ = _prompts(beh=a.beh)
    a_sys, a_pre = ANTI[a.beh]
    # Imported lazily: trajectory imports THIS module, so a top-level import would be
    # circular. By the time cmd_score runs, hf_transcripts is fully loaded and the cycle
    # resolves.
    import trajectory as TJ

    def _reframed(kind, convtxt):
        """A reframed context, built ONCE with an empty partial reply.

        A faithful version would rebuild the prompt at every token, since these templates
        quote the reply-so-far inside the request -- but that is the per-token cost that made
        the taskstep rollout arm take hours. Building it once, with the whole reply teacher-
        forced into the assistant slot, is the exact analogue of how the persona context is
        scored here, so the comparison is like-for-like. It does mean this measures the
        FRAMING, not the per-token re-injection, and it is the L=0 configuration -- the one
        where the reframing performed worst in the rollouts.
        """
        body = "(nothing yet)"
        ask = TJ.TASK_ASK[a.beh]
        if kind == "pivot":
            q = TJ.PIVOT_TMPL % (convtxt, body, ask)
        elif kind == "blunt":
            q = TJ.BLUNT_TMPL % (convtxt, body, ask)
        elif kind == "third":
            q = TJ.THIRD_TMPL % (convtxt, body, ask)
        else:
            q = TJ.TASK_TMPL % (convtxt, body, ask)
        return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True) + pre + TJ.COMPLY

    REFRAMED = ("task", "pivot", "third", "blunt")

    # --gen is whose transcripts these are, --model is who SCORES them. Conflating the two
    # makes cross-model scoring impossible: it would look for Llama transcripts under a Llama
    # directory that was never fetched, instead of scoring Qwen's transcripts with Llama.
    files = _transcripts(a.beh, a.gen or a.model, a.arm)
    if not files:
        print("no transcripts -- run fetch first")
        return
    out = []
    for i, f in enumerate(files):
        t = json.load(io.open(f, "r", encoding="utf-8"))
        pres = ((t.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        if pres is None:
            continue
        msgs = t["messages"]
        acc = {"t": [], "e": [], "d": [], "a": [], "da": []}
        for k_ in REFRAMED:
            acc[k_] = []
            acc["d" + k_] = []
        # ...and the same quantities restricted to the OPENING of each turn. The context gap
        # was measured earlier to live almost entirely in the first decile of a reply: both
        # contexts share the generated prefix, so once the turn has committed they agree on
        # how to continue. Pooling every token of every turn therefore dilutes a short strong
        # signal with a long flat tail. Collected here so the restriction costs no extra pass.
        head = {"t": [], "e": [], "d": [], "a": [], "da": []}
        for k_ in REFRAMED:
            head[k_] = []
            head["d" + k_] = []
        HEAD = 8
        # Every assistant turn is scored against its own preceding conversation, which is what
        # the decode conditions on. The behaviour score is per TRANSCRIPT, so the per-token
        # values are pooled across its turns rather than kept per turn.
        for j, m in enumerate(msgs):
            if m.get("role") != "assistant" or not (m.get("content") or "").strip():
                continue
            hist = msgs[:j]
            conv = [x for x in hist if x.get("role") != "system"]
            t_ctx = tok.apply_chat_template(hist, tokenize=False,
                                            add_generation_prompt=True) + pre
            e_msgs = ([{"role": "system", "content": e_sys}] if e_sys else []) + conv
            e_ctx = tok.apply_chat_template(e_msgs, tokenize=False,
                                            add_generation_prompt=True) + pre + e_pre
            a_msgs = [{"role": "system", "content": a_sys}] + conv
            a_ctx = tok.apply_chat_template(a_msgs, tokenize=False,
                                            add_generation_prompt=True) + pre + a_pre
            wt, tgt = _logit_window(model, tok, t_ctx, m["content"])
            if wt is None:
                continue
            we, _ = _logit_window(model, tok, e_ctx, m["content"])
            wa, _ = _logit_window(model, tok, a_ctx, m["content"])
            acc["t"] += _gather_lp(wt, tgt)
            acc["e"] += _gather_lp(we, tgt)
            acc["a"] += _gather_lp(wa, tgt)
            acc["d"] += _gather_lp(we - wt, tgt)      # the pure-difference distribution
            acc["da"] += _gather_lp(we - wa, tgt)
            for k, w in (("t", wt), ("e", we), ("a", wa),
                         ("d", we - wt), ("da", we - wa)):
                head[k] += _gather_lp(w, tgt)[:HEAD]
            convtxt = "\n".join("%s: %s" % (x["role"].upper(), x.get("content") or "")
                                 for x in conv)
            for k_ in REFRAMED:
                wr, _ = _logit_window(model, tok, _reframed(k_, convtxt), m["content"])
                lp_r = _gather_lp(wr, tgt)
                lp_d = _gather_lp(wr - wt, tgt)
                acc[k_] += lp_r
                acc["d" + k_] += lp_d
                head[k_] += lp_r[:HEAD]
                head["d" + k_] += lp_d[:HEAD]
                del wr
            del wt, we, wa
        if not acc["t"]:
            continue
        out.append({"file": os.path.basename(f), "presence": pres * 10.0,
                    "n_tok": len(acc["t"]),
                    "lp_target": st.mean(acc["t"]), "lp_elicited": st.mean(acc["e"]),
                    "lp_anti": st.mean(acc["a"]),
                    "delta": st.mean(acc["e"]) - st.mean(acc["t"]),
                    "diff": st.mean(acc["d"]), "diff_anti": st.mean(acc["da"]),
                    "h_lp_target": st.mean(head["t"]), "h_lp_elicited": st.mean(head["e"]),
                    "h_delta": st.mean(head["e"]) - st.mean(head["t"]),
                    "h_diff": st.mean(head["d"]), "h_diff_anti": st.mean(head["da"]),
                    **{("lp_" + k_): st.mean(acc[k_]) for k_ in REFRAMED},
                    **{("diff_" + k_): st.mean(acc["d" + k_]) for k_ in REFRAMED},
                    **{("h_lp_" + k_): st.mean(head[k_]) for k_ in REFRAMED},
                    **{("h_diff_" + k_): st.mean(head["d" + k_]) for k_ in REFRAMED}})
        if (i + 1) % 20 == 0:
            print("  scored %d/%d" % (i + 1, len(files)))
    os.makedirs(OUT, exist_ok=True)
    tag = a.model if not a.gen or a.gen == a.model else "%sgen_%ssc" % (a.gen, a.model)
    p = os.path.join(OUT, "hf_%s_%s_%s.jsonl" % (a.beh, a.arm, tag))
    with io.open(p, "w", encoding="utf-8", newline="") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("wrote %s (%d transcripts)" % (os.path.basename(p), len(out)))


def _auc(pos, neg):
    """P(a random positive scores above a random negative). 0.5 = chance, 1.0 = perfect.

    Reported instead of an accuracy because it needs no threshold: these scores have no
    natural zero the way the paired margins did, and picking a cutoff would be a free
    parameter fitted on the same data.
    """
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def cmd_report(a):
    for p in sorted(glob.glob(os.path.join(OUT, "hf_*.jsonl"))):
        rs = [json.loads(l) for l in io.open(p, "r", encoding="utf-8") if l.strip()]
        if not rs:
            continue
        name = os.path.basename(p)[3:-6]
        hi = [r for r in rs if r["presence"] >= a.hi]
        lo = [r for r in rs if r["presence"] <= a.lo]
        print("\n=== %s ===  n=%d   high(>=%g)=%d  low(<=%g)=%d"
              % (name, len(rs), a.hi, len(hi), a.lo, len(lo)))
        if not hi or not lo:
            print("  one class empty -- no discriminative test available here")
            continue
        for key, lab in (("lp_target", "target ctx alone"),
                         ("lp_elicited", "elicited ctx alone"),
                         ("lp_anti", "anti ctx alone"),
                         ("delta", "delta = elic - targ"),
                         ("diff", "PURE DIFF (e - t)"),
                         ("diff_anti", "PURE DIFF (e - a)"),
                         ("h_lp_target", "[head8] target alone"),
                         ("h_lp_elicited", "[head8] elicited alone"),
                         ("h_delta", "[head8] delta"),
                         ("h_diff", "[head8] PURE DIFF")) + tuple(
                             (k2, lab2) for k_ in ("task", "pivot", "third", "blunt")
                             for k2, lab2 in ((("lp_" + k_), (k_ + " alone")),
                                              (("diff_" + k_), (k_ + " PURE DIFF")),
                                              (("h_diff_" + k_), ("[head8] " + k_ + " DIFF")))):
            if key not in hi[0]:
                continue
            auc = _auc([r[key] for r in hi], [r[key] for r in lo])
            bar = "#" * int(round(abs(auc - 0.5) * 40))
            print("  %-22s AUC=%.3f  %s" % (lab, auc, bar))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in (("fetch", cmd_fetch), ("score", cmd_score)):
        p = sub.add_parser(n)
        p.set_defaults(fn=fn)
        p.add_argument("--beh", required=True)
        p.add_argument("--model", required=True, choices=sorted(MODELDIR))
        p.add_argument("--arm", required=True)
        p.add_argument("--gen", default=None,
                       help="model whose transcripts to score (default: same as --model)")
    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)
    p.add_argument("--hi", type=float, default=50.0)
    p.add_argument("--lo", type=float, default=20.0)
    args = ap.parse_args()
    args.fn(args)
