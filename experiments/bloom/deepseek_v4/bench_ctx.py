"""Batch ceiling at REAL 3-turn context length.

Short prompts make KV negligible, so a short-context sweep finds a ceiling that does
not exist for the real workload. This replays the actual transcripts from the vanilla
run: full context up to the last user message, then generates target_max_tokens=250,
and reports peak memory so the ceiling is the one that binds in a real round.
"""
import json, glob, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
D = "/workspace/bloom-wilt/experiments/bloom/runs_dsv4/self_harm/deepseek_v4_flash/vanilla_15s/round_1"
TARGET_MAX_TOKENS = 250

tok = AutoTokenizer.from_pretrained(REPO)
tok.chat_template = Path("/workspace/bloom-wilt/src/bloom/prompts/deepseek_v4_chat.jinja").read_text()
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

ctxs = []
for f in sorted(glob.glob(D + "/transcripts/*.json")):
    msgs = json.load(open(f))["messages"]
    while msgs and msgs[-1]["role"] != "user":   # cut back to the last user turn
        msgs = msgs[:-1]
    if msgs: ctxs.append(msgs)
lens = sorted(len(tok.apply_chat_template(c, add_generation_prompt=True,
              return_tensors="pt", return_dict=True)["input_ids"][0]) for c in ctxs)
print("transcripts: %d | ctx tokens min %d median %d max %d"
      % (len(lens), lens[0], lens[len(lens)//2], lens[-1]), flush=True)

m = AutoModelForCausalLM.from_pretrained(REPO, dtype="auto", device_map="auto",
                                         attn_implementation="eager").eval()
ndev = torch.cuda.device_count()
base = [torch.cuda.memory_allocated(i)/1e9 for i in range(ndev)]
tot_mem = [torch.cuda.get_device_properties(i).total_memory/1e9 for i in range(ndev)]
print("weights: %.1f GB | total VRAM: %.1f GB" % (sum(base), sum(tot_mem)), flush=True)

def run(b, n):
    sel = [ctxs[i % len(ctxs)] for i in range(b)]
    enc = tok([tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in sel],
              return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
    with torch.no_grad():
        m.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    return enc["input_ids"].shape[-1]

for b in (8, 15, 24, 32, 48, 64):
    try:
        run(b, 4)                                     # warm this batch geometry
        for i in range(ndev): torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.synchronize(); t = time.time()
        padded = run(b, TARGET_MAX_TOKENS)
        torch.cuda.synchronize(); dt = time.time() - t
        peak = sum(torch.cuda.max_memory_allocated(i)/1e9 for i in range(ndev))
        print("batch=%3d  padded_ctx=%4d  %6.1fs  %5.2f tok/s/seq  %6.1f tok/s total  "
              "peak %.1f GB  KV %.1f GB  free %.1f GB"
              % (b, padded, dt, TARGET_MAX_TOKENS/dt, b*TARGET_MAX_TOKENS/dt,
                 peak, peak - sum(base), sum(tot_mem) - peak), flush=True)
    except torch.cuda.OutOfMemoryError:
        print("batch=%3d  OOM" % b, flush=True)
        torch.cuda.empty_cache(); break
print("CTXBENCH DONE", flush=True)
