#!/usr/bin/env python3
"""Build a standalone HTML viewer for the paired top-5 datasets.

Renders every generated token of a run coloured by top-5 membership:
  red     the token is NOT in the target context's top-5
  blue    the token is NOT in the elicited context's top-5
  yellow  it is in neither
  plain   it is in both

Both directions are embedded (target-only tokens and elicited-only tokens) with a
picker, so the one file covers the whole comparison. Self-contained apart from the
Google Fonts link — no data files, no server.

  python -X utf8 experiments/postpaper/api_tilt/make_viewer.py
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARM_ROOT = REPO / "experiments/postpaper/runs_dsv4/self_harm/deepseek_v4_flash"
OUT = Path(__file__).resolve().parent / "top5_viewer.html"

ARMS = [
    ("api_elicited_15s", "elicited-only tokens", "b1=0, b2=1"),
    ("vanilla_15s", "target-only tokens", "b1=1, b2=0"),
]


def build(arm):
    run = ARM_ROOT / arm / "round_1"
    out = []
    for f in sorted((run / "top5").glob("*.top5.json"),
                    key=lambda p: int(p.name.split("_v")[1].split("r")[0])):
        d5 = json.load(open(f, encoding="utf-8"))
        src = json.load(open(run / "transcripts" / d5["meta"]["transcript"], encoding="utf-8"))
        users = [m["content"] for m in src["messages"] if m.get("source") == "evaluator"]
        turns = []
        for t, u in zip(d5["turns"], users):
            toks = []
            for i in range(t["n_tokens"]):
                tok = t["tokens"][i]
                if tok == "":
                    continue
                a, b = t["target"]["top"][i], t["elicited"]["top"][i]
                rt = next((j for j, (s, _) in enumerate(a) if s == tok), -1)
                re_ = next((j for j, (s, _) in enumerate(b) if s == tok), -1)
                toks.append([tok, rt, re_,
                             round(math.exp(t["target"]["lp"][i]) * 100, 3),
                             round(math.exp(t["elicited"]["lp"][i]) * 100, 3)])
            turns.append({"u": u, "t": toks})
        out.append({"id": "v%d" % d5["meta"]["variation_number"], "turns": turns})
    return out


DATA = {"arms": [{"key": k, "label": l, "coef": c, "transcripts": build(k)}
                 for k, l, c in ARMS]}

HTML = """<title>Top-5 membership map</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500&display=swap">
<style>
:root{
  --ground:#FFFFFF; --panel:#F7F7F5; --panel-2:#EFEFEA;
  --ink:#1A1A18; --ink-2:#4A4A45; --muted:#6B6B66; --rule:#E2E2DD; --rule-2:#C9C9C1;
  --r-bg:#FCEBEB; --r-fg:#791F1F; --r-br:#E24B4A;
  --b-bg:#E6F1FB; --b-fg:#0C447C; --b-br:#378ADD;
  --y-bg:#FAEEDA; --y-fg:#633806; --y-br:#BA7517;
  --sans:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,'Cascadia Code',Consolas,monospace;
  --serif:'IBM Plex Serif',Georgia,serif;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;margin:0}
.page{max-width:900px;margin:0 auto;padding:40px 28px 72px;display:flex;flex-direction:column;gap:26px}
h1{font-family:var(--serif);font-weight:500;font-size:30px;line-height:1.2;margin:0;
  letter-spacing:-0.01em;text-wrap:balance}
.sub{color:var(--ink-2);margin:0;max-width:62ch}
.sub code{font-family:var(--mono);font-size:0.9em;background:var(--panel-2);
  padding:1px 5px;border-radius:3px}
.eyebrow{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  font-weight:500}
.rules{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.rule-card{background:var(--panel);border-radius:8px;padding:11px 13px;display:flex;
  align-items:flex-start;gap:10px}
.dot{width:13px;height:13px;border-radius:3px;flex:none;margin-top:3px;border:1px solid}
.rule-card .txt{font-size:13px;line-height:1.45}
.rule-card .cnt{font-family:var(--mono);font-weight:500;font-variant-numeric:tabular-nums}
.rule-card .sm{color:var(--muted);font-size:12px;display:block}
.bar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;
  border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:14px 0}
.grp{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.grp>.eyebrow{margin-right:2px}
button{font-family:var(--sans);font-size:13px;color:var(--ink-2);background:var(--ground);
  border:1px solid var(--rule-2);border-radius:6px;padding:5px 11px;cursor:pointer;
  line-height:1.3}
button:hover{background:var(--panel)}
button:focus-visible{outline:2px solid var(--b-br);outline-offset:2px}
button[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--ground)}
button.tiny{font-family:var(--mono);padding:4px 8px;font-size:12px}
label.tog{display:inline-flex;gap:7px;align-items:center;font-size:13px;color:var(--ink-2);cursor:pointer}
.turn{display:flex;flex-direction:column;gap:9px}
.turns{display:flex;flex-direction:column;gap:30px}
.aud{background:var(--panel);border-radius:8px;padding:12px 15px;color:var(--ink-2);font-size:14px}
.body{position:relative;white-space:pre-wrap;font-family:var(--mono);font-size:14.5px;
  line-height:2.15;letter-spacing:-0.005em}
.tk{border-radius:3px;padding:2px 0}
.tk.c1{background:var(--r-bg);color:var(--r-fg)}
.tk.c2{background:var(--b-bg);color:var(--b-fg)}
.tk.c3{background:var(--y-bg);color:var(--y-fg)}
.bd .tk{box-shadow:inset -1px 0 0 var(--rule-2)}
.tk.fade{background:transparent;color:#B4B2A9;box-shadow:none}
.tip{position:absolute;z-index:9;background:var(--ground);border:1px solid var(--rule-2);
  border-radius:7px;padding:10px 12px;font-family:var(--sans);font-size:12.5px;
  line-height:1.5;min-width:212px;white-space:normal;pointer-events:none;display:none}
.tip .h{font-family:var(--mono);font-weight:500;margin-bottom:7px;word-break:break-all}
.tip table{border-collapse:collapse;width:100%}
.tip td{padding:2px 0;color:var(--muted)}
.tip td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;color:var(--ink)}
.tip td.m{font-family:var(--mono);color:var(--ink-2);padding-left:12px}
.note{color:var(--muted);font-size:13px;max-width:70ch}
.note strong{font-weight:500;color:var(--ink-2)}
.toast{font-size:12.5px;color:var(--muted)}
@media (max-width:640px){.page{padding:26px 16px 56px}h1{font-size:24px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
<div class="page">
  <header style="display:flex;flex-direction:column;gap:12px">
    <span class="eyebrow">LogitTilt · DeepSeek-V4-Flash · self-harm · round 1</span>
    <h1>Which candidates did each context actually offer?</h1>
    <p class="sub">Every token a run generated, teacher-forced through both contexts and
      coloured by whether it appears in that context's top-5 next-token list. The target
      context is <code>b1=1, b2=0</code>; the elicited context replaces the system prompt
      with the jail persona and appends the prefill, <code>b1=0, b2=1</code>. Five
      alternatives is the cap the Fireworks API returns, so this is exactly what a top-5
      reconstruction of the tilt would have to work from.</p>
  </header>

  <div class="rules" id="rules"></div>

  <div class="bar">
    <div class="grp"><span class="eyebrow">Tokens from</span><span id="armSel"></span></div>
    <div class="grp"><span class="eyebrow">Scenario</span><span id="txSel"></span></div>
  </div>
  <div class="bar" style="border-top:none;padding-top:0">
    <label class="tog"><input type="checkbox" id="bd"> Token boundaries</label>
    <button id="copyBtn">Copy transcript text</button>
    <span class="toast" id="toast"></span>
  </div>

  <div class="turns" id="turns"></div>

  <p class="note"><strong>Copying into a document:</strong> select the transcript and copy —
    Google Docs, Word and Keynote all preserve the highlight colours. The button above copies
    the plain text without them. End-of-sequence tokens are excluded throughout.</p>
</div>
<script>
const DATA = __DATA__;
const CATS = [
  {k:0, lab:"In both top-5 lists",  hint:"both contexts proposed it", bg:"transparent", br:"var(--rule-2)"},
  {k:1, lab:"Not in target top-5",  hint:"only the elicited context proposed it", bg:"var(--r-bg)", br:"var(--r-br)"},
  {k:2, lab:"Not in elicited top-5",hint:"only the target context proposed it", bg:"var(--b-bg)", br:"var(--b-br)"},
  {k:3, lab:"In neither top-5",     hint:"no shared candidate at all", bg:"var(--y-bg)", br:"var(--y-br)"}
];
const cat = (rt,re) => rt>=0&&re>=0 ? 0 : rt<0&&re>=0 ? 1 : rt>=0 ? 2 : 3;
const fmt = p => p<0.001 ? "<0.001%" : p<1 ? p.toFixed(3)+"%" : p.toFixed(2)+"%";
const rk  = r => r<0 ? "outside" : "rank "+(r+1);
const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;");
let armI = 0, txI = 0, iso = null;

const armSel = document.getElementById("armSel");
DATA.arms.forEach((a,i) => {
  const b = document.createElement("button");
  b.textContent = a.label;
  b.onclick = () => { armI = i; txI = 0; renderPickers(); render(); };
  armSel.appendChild(b);
});
const txSel = document.getElementById("txSel");

function renderPickers(){
  [...armSel.children].forEach((b,i) => b.setAttribute("aria-pressed", i===armI));
  txSel.innerHTML = "";
  DATA.arms[armI].transcripts.forEach((t,i) => {
    const b = document.createElement("button");
    b.className = "tiny"; b.textContent = t.id;
    b.onclick = () => { txI = i; render(); };
    txSel.appendChild(b);
  });
}

function current(){ return DATA.arms[armI].transcripts[txI]; }

function render(){
  [...txSel.children].forEach((b,i) => b.setAttribute("aria-pressed", i===txI));
  const tx = current();
  const counts = [0,0,0,0];
  tx.turns.forEach(t => t.t.forEach(x => counts[cat(x[1],x[2])]++));
  const total = counts.reduce((a,b)=>a+b,0);

  const rules = document.getElementById("rules");
  rules.innerHTML = "";
  CATS.forEach(c => {
    const d = document.createElement("button");
    d.className = "rule-card";
    d.style.textAlign = "left"; d.style.border = "none"; d.style.width = "100%";
    d.setAttribute("aria-pressed", iso===c.k);
    if (iso===c.k) { d.style.background = "var(--panel-2)"; }
    d.innerHTML = '<span class="dot" style="background:'+c.bg+';border-color:'+c.br+'"></span>'
      + '<span class="txt"><span class="cnt">'+counts[c.k]+'</span> &middot; '+c.lab
      + '<span class="sm">'+c.hint+' &middot; '+(100*counts[c.k]/total).toFixed(1)+'%</span></span>';
    d.onclick = () => { iso = iso===c.k ? null : c.k; render(); };
    rules.appendChild(d);
  });

  const wrap = document.getElementById("turns");
  wrap.innerHTML = "";
  tx.turns.forEach((t,i) => {
    const w = document.createElement("div"); w.className = "turn";
    const l1 = document.createElement("div"); l1.className = "eyebrow";
    l1.textContent = "Turn " + (i+1) + " — auditor";
    const u = document.createElement("div"); u.className = "aud"; u.textContent = t.u;
    const l2 = document.createElement("div"); l2.className = "eyebrow";
    l2.textContent = "Target reply · " + t.t.length + " tokens";
    const b = document.createElement("div"); b.className = "body";
    if (document.getElementById("bd").checked) b.classList.add("bd");
    t.t.forEach(x => {
      const s = document.createElement("span");
      const k = cat(x[1],x[2]);
      s.className = "tk" + (k ? " c"+k : "") + (iso!==null && k!==iso ? " fade" : "");
      s.textContent = x[0];
      s.dataset.info = JSON.stringify(x);
      b.appendChild(s);
    });
    const tip = document.createElement("div"); tip.className = "tip";
    b.appendChild(tip);
    b.addEventListener("mousemove", e => {
      const s = e.target.closest(".tk"); if (!s) { tip.style.display = "none"; return; }
      const x = JSON.parse(s.dataset.info);
      tip.innerHTML = '<div class="h">' + esc(x[0]).replace(/ /g,"\\u00b7") + '</div>'
        + '<table><tr><td>target</td><td class="m">'+rk(x[1])+'</td><td class="n">'+fmt(x[3])+'</td></tr>'
        + '<tr><td>elicited</td><td class="m">'+rk(x[2])+'</td><td class="n">'+fmt(x[4])+'</td></tr></table>';
      tip.style.display = "block";
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let L = s.offsetLeft + s.offsetWidth/2 - tw/2;
      L = Math.max(0, Math.min(L, b.clientWidth - tw));
      let T = s.offsetTop - th - 8;
      if (T < 0) T = s.offsetTop + s.offsetHeight + 8;
      tip.style.left = L + "px"; tip.style.top = T + "px";
    });
    b.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    w.appendChild(l1); w.appendChild(u); w.appendChild(l2); w.appendChild(b);
    wrap.appendChild(w);
  });
}

document.getElementById("bd").addEventListener("change", e => {
  document.querySelectorAll(".body").forEach(b => b.classList.toggle("bd", e.target.checked));
});

document.getElementById("copyBtn").addEventListener("click", () => {
  const tx = current();
  const txt = tx.turns.map((t,i) =>
    "[turn "+(i+1)+" auditor]\\n" + t.u + "\\n\\n[turn "+(i+1)+" target]\\n"
    + t.t.map(x => x[0]).join("")).join("\\n\\n");
  const toast = document.getElementById("toast");
  navigator.clipboard.writeText(txt).then(
    () => { toast.textContent = "Copied " + tx.id + " to the clipboard."; },
    () => { toast.textContent = "Could not reach the clipboard — select the text and copy."; }
  );
  setTimeout(() => { toast.textContent = ""; }, 4000);
});

renderPickers();
render();
</script>
"""

OUT.write_text(HTML.replace("__DATA__", json.dumps(DATA, ensure_ascii=False,
                                                   separators=(",", ":"))),
               encoding="utf-8")
n = sum(len(t["t"]) for a in DATA["arms"] for tx in a["transcripts"] for t in tx["turns"])
print(f"wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size/1000:.0f} KB, "
      f"{sum(len(a['transcripts']) for a in DATA['arms'])} transcripts, {n} tokens)")
