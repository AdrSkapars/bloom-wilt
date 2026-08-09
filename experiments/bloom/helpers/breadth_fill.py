import os, sys, glob, json, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import method_tuning as M

RF = "/workspace/inversion_optimisation/experiments/bloom/runs_final"
BEH = ["racial", "political", "delusions", "self_harm", "medical", "deception", "selfpres", "goblin"]
MOD = ["Qwen_Qwen3.5-4B", "google_gemma-4-e4b-it",
       "meta-llama_Llama-3.2-3B-Instruct", "microsoft_Phi-4-mini-instruct"]

def rd(a):
    tf, sc, rnd = a; r = M._read((tf, sc))
    if r: r["round"] = rnd
    return r
def load_rounds(folder, rmax=None):
    tasks = []
    for jp in sorted(glob.glob(os.path.join(folder, "round_*", "judgment.json"))):
        rnd = int(os.path.basename(os.path.dirname(jp)).split("_")[1])
        if rmax and rnd > rmax:
            continue
        j = json.load(open(jp, encoding="utf-8"))
        sc = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
              if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(os.path.dirname(jp), "transcripts", "*.json")):
            tasks.append((tf, sc, rnd))
    return [r for r in M.POOL.map(rd, tasks) if r]
def band(pts, x_bon):
    fr = M._frontier(pts); b = [f for f in fr if f[0] >= x_bon]
    pick = max(b, key=lambda f: f[1]) if b else max(fr, key=lambda f: (f[0], f[1]))
    return pick[2]
def agg(chosen):
    return st.mean(c["score"] for c in chosen) * 10, st.mean(c["mean"] for c in chosen)   # presence, arith
def subdir(cell, pat):
    for p in sorted(glob.glob(os.path.join(cell, pat))):
        if os.path.isdir(p): return p
    return None

for beh in BEH:
    for mod in MOD:
        cell = os.path.join(RF, beh, mod)
        bon = load_rounds(os.path.join(cell, "bon"))
        bfr = M._frontier(bon); bpick = max(bfr, key=lambda f: f[1])
        x_bon = bpick[0]; van = agg(bpick[2])
        jd = subdir(cell, "jail_b*")
        lt = agg(band(load_rounds(jd), x_bon)) if jd else van          # jail=None (beta 0) -> vanilla
        cd = subdir(cell, "combo/beta_*")
        wilt = agg(band(load_rounds(cd, 5), x_bon)) if cd else None     # WILT truncated to 5 rounds
        f = lambda t: f"{t[0]:.1f}/{t[1]:.1f}" if t else "--"
        print(f"{beh}|{mod.split('_')[0]}| V {f(van)} | LT {f(lt)} | W {f(wilt)}  (x_bon={x_bon:.1f}{' LT=van' if not jd else ''})", flush=True)
M.POOL.shutdown(wait=False)
print("DONE", flush=True)
