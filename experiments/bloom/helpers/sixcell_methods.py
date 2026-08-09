#!/usr/bin/env python3
"""6-cell (model x behaviour) method overview, colour-blind-friendly.
Per cell: post-hoc Pareto CURVE for BoN / LogitTilt(jail) / G-PAIR / WILT(combo) (each with a
best-in-band marker), single POINT for BEAST-in/-out / FLRT, grey >=BoN-plausibility band.
CB-safe: Okabe-Ito qualitative palette + distinct MARKER per method + distinct LINESTYLE per curve
(so methods separate by shape/dash even in greyscale). Reads the frontier cache (sixcell_cache.json)
if present, else rebuilds from runs_final. Run on the box: UV_NO_SYNC=1 uv run --no-sync python -X utf8 sixcell_methods.py
"""
import sys, os, glob, json, statistics as st
from concurrent.futures import ThreadPoolExecutor
REPO = "/workspace/inversion_optimisation"
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(REPO, "experiments/bloom/helpers"))
import pareto_analysis as PA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RF = os.path.join(REPO, "experiments/bloom/runs_final")
FIGDIR = os.path.join(REPO, "paper/figures")
BEHS = [("self-harm", "self_harm"), ("deception", "deception"), ("political", "political")]
MODELS = [("Qwen3.5-4B", "Qwen_Qwen3.5-4B"), ("Gemma-4-E4B", "google_gemma-4-e4b-it")]
POOL = ThreadPoolExecutor(max_workers=24)


def _read(args):
    tf, score = args
    try:
        d = json.load(open(tf, encoding="utf-8"))
        v = d.get("metadata", {}).get("variation_number")
        if v is None or v not in score:
            return None
        ps = d.get("prob_stats")
        if ps and ps.get("n"):
            prob = ps["mean"]
        else:
            tm = [m for m in d["messages"] if m.get("source") == "target"]
            num = sum(m["prob_stats"]["n"] * m["prob_stats"]["mean"] for m in tm if m.get("prob_stats"))
            den = sum(m["prob_stats"]["n"] for m in tm if m.get("prob_stats"))
            if not den:
                return None
            prob = num / den
        return {"scenario": v, "prob": prob, "score": score[v]}
    except Exception:
        return None


def extract2(run_dir, rmax=None):
    tasks = []
    for jp in sorted(glob.glob(os.path.join(run_dir, "round_*", "judgment.json"))):
        if rmax is not None and int(os.path.basename(os.path.dirname(jp)).split("_")[1]) > rmax:
            continue
        j = json.load(open(jp, encoding="utf-8"))
        score = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
                 if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(os.path.dirname(jp), "transcripts", "*.json")):
            tasks.append((tf, score))
    return [r for r in POOL.map(_read, tasks) if r]


def op_point(folder):
    g = {}
    for p in extract2(folder):
        g.setdefault(p["scenario"], []).append(p)
    if not g:
        return None
    best = [max(v, key=lambda p: p["score"]) for v in g.values()]
    return [st.mean(b["prob"] for b in best), st.mean(b["score"] for b in best)]


def frontier(folder, rmax=None):
    pts = extract2(folder, rmax)
    return PA.pareto_frontier(pts) if pts else None


def find(cell, *cands):
    for c in cands:
        hits = sorted(glob.glob(os.path.join(cell, c)))
        if hits:
            return hits[0]
    return None


# ---- build cache incrementally (resume-safe); reuse if present ----
CACHE = os.path.join(HERE, "sixcell_cache.json")
C = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
for mlab, mdir in MODELS:
    for blab, bdir in BEHS:
        key = f"{mlab}|{blab}"
        if key in C:
            continue
        cell = os.path.join(RF, bdir, mdir)
        jail = find(cell, "jail_b*"); combo = find(cell, "combo/beta_*"); tb = find(cell, "tokbias_tuned", "tokbias")
        res = {}
        curves = {"BoN": os.path.join(cell, "bon"), "LogitTilt": jail,
                  "G-PAIR": os.path.join(cell, "gpair_t3_sfull"), "WILT": combo}
        points = {"BEAST-in": os.path.join(cell, "input_search_3turn"), "BEAST-out": os.path.join(cell, "output_search_3turn"),
                  "FLRT": os.path.join(cell, "flrt"), "TokenBias": tb}
        for name, fol in curves.items():
            try:
                res[name] = {"curve": frontier(fol, rmax=5 if name == "WILT" else None)} if fol else None
            except Exception as e:
                res[name] = None; print("ERR", key, name, repr(e), flush=True)
        for name, fol in points.items():
            try:
                res[name] = {"pt": op_point(fol)} if fol else None
            except Exception as e:
                res[name] = None; print("ERR", key, name, repr(e), flush=True)
        C[key] = res
        json.dump(C, open(CACHE, "w", encoding="utf-8"))
        print("cached", key, flush=True)

# ---- plot 2 rows (models) x 3 cols (behaviours), CB-safe ----
SZ = 78
# Okabe-Ito colour-blind-safe palette; method -> (colour, curve linestyle or None, marker shape)
STYLE = {
    "BoN":       ("#000000", ":",  "o"),   # black
    "LogitTilt": ("#0072B2", "-",  "*"),   # blue
    "G-PAIR":    ("#009E73", "--", "P"),   # bluish green
    "WILT":      ("#D55E00", "-.", "X"),   # vermillion
    "BEAST-in":  ("#56B4E9", None, "s"),   # sky blue
    "BEAST-out": ("#E69F00", None, "^"),   # orange
    "FLRT":      ("#CC79A7", None, "D"),   # reddish purple
}
CURVES = ["BoN", "LogitTilt", "G-PAIR", "WILT"]
POINTS = ["BEAST-in", "BEAST-out", "FLRT"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=140)
for r, (mlab, _) in enumerate(MODELS):
    for c, (blab, _) in enumerate(BEHS):
        ax = axes[r][c]; cell = C.get(f"{mlab}|{blab}", {}); xs = []
        bon_curve = (cell.get("BoN") or {}).get("curve")
        x_bon = max(bon_curve, key=lambda p: p[1])[0] if bon_curve else None
        if x_bon is not None:
            ax.axvspan(x_bon, 1e4, color="#bcbcbc", alpha=0.22, lw=0, zorder=0)   # grey >=BoN band
        for name in CURVES:
            e = cell.get(name)
            if not e or not e.get("curve"):
                continue
            col, ls, mk = STYLE[name]; cur = e["curve"]; xs += [p[0] for p in cur]
            ax.plot([p[0] for p in cur], [p[1] * 10 for p in cur], ls, color=col, lw=2.1, zorder=3)
            band = [p for p in cur if x_bon is not None and p[0] >= x_bon]
            bp = max(band, key=lambda p: p[1]) if band else max(cur, key=lambda p: (p[0], p[1]))
            ax.scatter([bp[0]], [bp[1] * 10], s=SZ, marker=mk, color=col, edgecolor="#111111", lw=1.0, zorder=8)
        for name in POINTS:
            e = cell.get(name)
            if not e or not e.get("pt"):
                continue
            col, ls, mk = STYLE[name]; xs.append(e["pt"][0])
            ax.scatter([e["pt"][0]], [e["pt"][1] * 10], s=SZ, marker=mk, color=col, edgecolor="#111111", lw=1.0, zorder=7)
        if xs:
            ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_title(f"{mlab} · {blab.capitalize()}", fontsize=16)
        ax.set_ylim(0, 103)
        ax.grid(True, color="#e6e5de", lw=0.5, zorder=1); ax.tick_params(labelsize=13)
        if r == 1:
            ax.set_xlabel("Output probability (%)", fontsize=16)
        if c == 0:
            ax.set_ylabel("Behaviour score (%)", fontsize=16)
handles = []
for name in CURVES + POINTS:
    col, ls, mk = STYLE[name]
    handles.append(Line2D([0], [0], color=col, ls=(ls or "None"), lw=2.4,
                          marker=mk, markersize=12, markeredgecolor="#111111", label=name))
fig.tight_layout(rect=[0, 0, 0.85, 0.99])
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.855, 0.5), frameon=False,
           fontsize=15, title="Method", title_fontsize=15)
fig.savefig(os.path.join(FIGDIR, "pareto_6cell.pdf"), facecolor="white")
fig.savefig(os.path.join(HERE, "sixcell_methods.png"), facecolor="white")
print("saved paper/figures/pareto_6cell.pdf", flush=True)
POOL.shutdown(wait=False)
