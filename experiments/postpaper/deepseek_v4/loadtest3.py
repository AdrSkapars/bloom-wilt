import time, os, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
TMPL = "/workspace/bloom-wilt/src/bloom/prompts/deepseek_v4_chat.jinja"

tok = AutoTokenizer.from_pretrained(REPO)
print("tokenizer ok | ships chat_template:", bool(getattr(tok, "chat_template", None)), flush=True)
if not getattr(tok, "chat_template", None):
    tok.chat_template = Path(TMPL).read_text(encoding="utf-8")
    print("attached jinja from prompts/", flush=True)

t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(
    REPO, dtype="auto", device_map="auto", attn_implementation="eager"
).eval()
print("LOADED in %.1f min" % ((time.time() - t0) / 60), flush=True)

qc = getattr(m.config, "quantization_config", None)
print("quant_method  :", getattr(qc, "quant_method", None) or qc)
mods = {type(mm).__name__ for _, mm in m.named_modules()}
print("quant modules :", sorted(x for x in mods if "FP8" in x or "FP4" in x or "Quant" in x))
w = m.model.layers[3].mlp.experts
for attr in ("gate_up_proj", "down_proj"):
    p = getattr(w, attr, None)
    if p is not None:
        print("  experts.%s: %s %s" % (attr, tuple(p.shape), p.dtype))
tot = 0
for i in range(torch.cuda.device_count()):
    g = torch.cuda.memory_allocated(i) / 1e9; tot += g
    print("  gpu%d %.1f GB" % (i, g))
print("resident total: %.1f GB" % tot, flush=True)
print("devices spanned:", len(set(str(v) for v in (m.hf_device_map or {}).values())), flush=True)

msgs = [{"role": "user", "content": "Name three colours, then count from 1 to 5."}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = enc["input_ids"].to(m.device)
print("prompt tokens:", ids.shape[-1], "| rendered:", repr(tok.decode(ids[0]))[:200], flush=True)
t1 = time.time()
with torch.no_grad():
    out = m.generate(ids, max_new_tokens=60, do_sample=False)
dt = time.time() - t1
new = out[0][ids.shape[-1]:]
print("GENERATED %d tok in %.1fs (%.2f tok/s)" % (len(new), dt, len(new) / dt), flush=True)
print("SAMPLE >>>", repr(tok.decode(new, skip_special_tokens=True)), flush=True)
