"""[jail engine=api_tilt] Run the WILT target through a hosted /completions API.

WHY THIS EXISTS
---------------
LogitTilt samples from ``z = b1*l_target + b2*l_jail`` — two contexts of the SAME weights
stepped in lockstep, mixed in full-vocab logit space before sampling. No hosted API exposes
that, so the local `hf_full` engine needs the weights on a GPU.

Two corners of the (b1, b2) plane need NO mixing and so ARE exactly reproducible over a
text API, because in both the sampling distribution is a single context's own softmax:

    b1=1, b2=0   target only     z = l_target    (vanilla / BoN)
    b1=0, b2=1   elicited only   z = l_jail      (the jail context alone)

Everything in between is refused rather than silently approximated — a top-k approximation
to the true tilt is a separate, explicit experiment.

The two corners still differ in how the reported plausibility is obtained:

  * target only   — the tokens were drawn from the target distribution, so the API's own
                    generation logprobs ARE the on-policy target probs. One call.
  * elicited only — the tokens were drawn from the JAIL context, but the metric we report
                    (matching `_jail_generate_hf`'s `best_token_probs`) is the UNMODIFIED
                    TARGET probability of those tokens. That needs a teacher-forced pass
                    over the TARGET context: `/completions` with ``echo=true`` and
                    ``max_tokens=0`` scores a supplied sequence, and the sequence is sent as
                    TOKEN IDS so the sampled tokenization survives (see `score_ids`).

Validated against local ground truth (see deepseek_v4/RUNBOOK.md): re-scoring the local
vanilla run's stored `gen_token_ids` through this template reproduced every token id
exactly and matched the stored per-token probabilities to ~1.3 pp on the mean, running
consistently a little lower (serving precision differs from the local FP4 load).

WHAT IS NOT SUPPORTED (each raises rather than degrading quietly)
  * any b1/b2 mix other than the two corners above
  * `target_floor` > 0 — the floor masks on the TRUE TARGET distribution at every step,
    which needs target logits while sampling
  * `b3` (negative steering) and `tokbias` — both are logit-space edits
  * a jail model distinct from the target — the API serves one model per request

PROMPT RENDERING
----------------
The API takes a string, so prompts are rendered HERE with the same Jinja chat template the
local path attaches to the tokenizer (`BLOOM_TARGET_CHAT_TEMPLATE`). That is what makes the
tokenization identical to the local run; the ids are verified against the returned
`token_ids`, so a template drift fails loudly instead of shifting the distribution.
"""
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from . import core

# Hosted providers whose /completions endpoint does BOTH generation-with-logprobs and
# echo scoring. Verified empirically (scratchpad probe): Fireworks returns tokens,
# token_ids, token_logprobs and text_offset for echo=true + max_tokens=0.
_PROVIDERS = {
    "fireworks": {"base": "https://api.fireworks.ai/inference/v1", "key_env": "FIREWORKS_API_KEY"},
    "together":  {"base": "https://api.together.xyz/v1",           "key_env": "TOGETHER_API_KEY"},
}

# DeepSeek-V4's BOS. The template takes it as a variable (transformers passes
# tokenizer.bos_token); with no tokenizer here it comes from BLOOM_TARGET_BOS_TOKEN.
_DEFAULT_BOS = "<｜begin▁of▁sentence｜>"


class ApiTiltTarget:
    """One hosted model, addressed through /completions for both generation and scoring."""

    def __init__(self, model: str, provider: str = "fireworks",
                 template_path: str = "", bos_token: str = _DEFAULT_BOS,
                 timeout: float = 300.0, max_retries: int = 5):
        if provider not in _PROVIDERS:
            raise RuntimeError(
                f"BLOOM_TARGET_API={provider!r} unknown; known providers: {sorted(_PROVIDERS)}")
        p = _PROVIDERS[provider]
        key = (os.environ.get(p["key_env"], "") or "").strip()
        if not key:
            raise RuntimeError(
                f"{p['key_env']} is not set — the api_tilt engine needs it to reach {provider}.")
        self.model, self.provider, self.base = model, provider, p["base"]
        self.timeout, self.max_retries = timeout, max_retries
        # Cloudflare in front of some providers rejects the default urllib User-Agent.
        self._hdr = {"User-Agent": "curl/8.4.0", "Content-Type": "application/json",
                     "Accept": "application/json", "Authorization": f"Bearer {key}"}
        if not template_path:
            raise RuntimeError(
                "api_tilt needs BLOOM_TARGET_CHAT_TEMPLATE — prompts are rendered here, not by "
                "a tokenizer, so the template is the only thing pinning the API's tokenization "
                "to the local run's.")
        from jinja2 import Environment
        self._tpl = Environment().from_string(Path(template_path).read_text(encoding="utf-8"))
        self._bos = bos_token
        self.n_calls = 0
        self.n_prompt_tokens = 0
        self.n_gen_tokens = 0

    # ── prompt rendering ────────────────────────────────────────────────────────────
    def render(self, msgs: List[Dict], add_generation_prompt: bool = True) -> str:
        """Chat-template render, mirroring tokenizer.apply_chat_template(tokenize=False)."""
        return self._tpl.render(messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
                                add_generation_prompt=add_generation_prompt,
                                bos_token=self._bos)

    # ── HTTP ────────────────────────────────────────────────────────────────────────
    def _post(self, body: Dict) -> Dict:
        data = json.dumps(body).encode()
        last = ""
        for attempt in range(self.max_retries):
            req = urllib.request.Request(self.base + "/completions", data=data, headers=self._hdr)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.n_calls += 1
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode()[:300]}"
                # 4xx other than rate-limit is a request bug — retrying just burns time.
                if e.code not in (408, 409, 429) and e.code < 500:
                    raise RuntimeError(f"api_tilt request rejected — {last}")
            except Exception as e:                      # timeout / connection reset
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(30.0, 2.0 ** attempt) * (0.5 + random.random()))
        raise RuntimeError(f"api_tilt request failed after {self.max_retries} attempts — {last}")

    # ── the two primitives ──────────────────────────────────────────────────────────
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict:
        """Sample a continuation. Returns text + the sampled ids and their own logprobs.

        top_p/top_k are pinned wide open so this is plain temperature sampling from the
        full distribution, exactly like the local `torch.multinomial(softmax(z/T))` step —
        a provider-side default top_k would silently truncate the tail.
        """
        r = self._post({"model": self.model, "prompt": prompt, "max_tokens": int(max_tokens),
                        "temperature": float(temperature), "top_p": 1.0, "top_k": 0,
                        "logprobs": 1, "echo": False})
        ch = r["choices"][0]
        lp = ch.get("logprobs") or {}
        u = r.get("usage") or {}
        self.n_prompt_tokens += int(u.get("prompt_tokens") or 0)
        self.n_gen_tokens += int(u.get("completion_tokens") or 0)
        return {"text": ch.get("text") or "",
                "ids": list(lp.get("token_ids") or []),
                "logprobs": [v for v in (lp.get("token_logprobs") or [])],
                "finish_reason": ch.get("finish_reason")}

    def _echo(self, prompt, logprobs: int = 1) -> Dict:
        """echo=true + max_tokens=0 — teacher-forced scoring of a supplied sequence.
        `prompt` is a string or a list of token ids. `logprobs` also asks for that many
        ALTERNATIVES per position (Fireworks caps it at 5)."""
        r = self._post({"model": self.model, "prompt": prompt,
                        "max_tokens": 0, "echo": True, "logprobs": int(logprobs)})
        self.n_prompt_tokens += int((r.get("usage") or {}).get("prompt_tokens") or 0)
        return r["choices"][0].get("logprobs") or {}

    def prefix_ids(self, prefix: str) -> List[int]:
        """Token ids of a rendered prompt, from the provider's own tokenizer."""
        return list(self._echo(prefix).get("token_ids") or [])

    def score_ids(self, prefix: str, cont_ids: List[int]) -> List[float]:
        """Teacher-forced logprobs of the EXACT sampled tokens `cont_ids`, given `prefix`.

        Scoring the continuation as TEXT would be wrong: sampling can produce a
        non-canonical tokenization, so re-tokenizing the decoded string yields a different
        token sequence (observed: 73 sampled tokens re-tokenizing to 72). The per-token
        probabilities would then describe a sequence the model never actually emitted.

        So the prompt is sent as TOKEN IDS — `prefix_ids + cont_ids` — which the provider
        scores position-for-position. That reproduces the local engine's metric exactly: the
        unmodified-target probability of each token that was really sampled. Costs one extra
        (max_tokens=0) call to tokenize the prefix.

        Requires a provider whose /completions accepts an integer-array prompt (verified on
        Fireworks). A provider that rejects it surfaces as a request error, not silently.
        """
        n = len(cont_ids)
        if n == 0:
            return []
        pre = self.prefix_ids(prefix)
        lp = self._echo(list(pre) + list(cont_ids))
        ids, lps = list(lp.get("token_ids") or []), list(lp.get("token_logprobs") or [])
        if len(ids) != len(pre) + n or ids[-n:] != list(cont_ids):
            raise RuntimeError(
                f"api_tilt scoring did not echo the supplied token ids back unchanged "
                f"(sent {len(pre)}+{n}, got {len(ids)}). The provider is re-tokenizing an "
                f"id-array prompt; this engine needs one that does not.")
        return [float(v) for v in lps[-n:]]

    def score_ids_topk(self, prefix: str, cont_ids: List[int], top_k: int = 5) -> Dict:
        """Like `score_ids`, but also returns the top-`top_k` ALTERNATIVES at each position.

        `top_logprobs[i]` is the distribution that predicted token i, as a
        {token_string: logprob} dict — the provider returns no ids for the alternatives, so
        they are compared as strings. The sampled token is often NOT among its own top-k;
        that is the interesting case, not an error.

        Returns {"tokens", "lp", "top"} over the continuation only, where `top[j]` is a list
        of (token_string, logprob) sorted most-likely first.
        """
        n = len(cont_ids)
        if n == 0:
            return {"tokens": [], "lp": [], "top": []}
        pre = self.prefix_ids(prefix)
        lp = self._echo(list(pre) + list(cont_ids), logprobs=int(top_k))
        ids = list(lp.get("token_ids") or [])
        if len(ids) != len(pre) + n or ids[-n:] != list(cont_ids):
            raise RuntimeError(
                f"api_tilt top-k scoring did not echo the supplied token ids back unchanged "
                f"(sent {len(pre)}+{n}, got {len(ids)}).")
        tl = list(lp.get("top_logprobs") or [])
        top = [sorted((d or {}).items(), key=lambda kv: -kv[1]) for d in tl[-n:]]
        return {"tokens": list(lp.get("tokens") or [])[-n:],
                "lp": [float(v) for v in list(lp.get("token_logprobs") or [])[-n:]],
                "top": top}

    def stats(self) -> str:
        return (f"{self.n_calls} calls, {self.n_prompt_tokens} prompt tok, "
                f"{self.n_gen_tokens} generated tok")


def load_api_target(target_model_id: str) -> Dict:
    """Build the api_tilt handle. Mirrors `_load_hf_poe_models`'s return shape so the two
    engines are interchangeable at the call site (no_think wrappers included)."""
    model = target_model_id[len("api/"):] if target_model_id.startswith("api/") else target_model_id
    client = ApiTiltTarget(
        model=model,
        provider=(os.environ.get("BLOOM_TARGET_API", "") or "fireworks").strip(),
        template_path=(os.environ.get("BLOOM_TARGET_CHAT_TEMPLATE", "") or "").strip(),
        bos_token=(os.environ.get("BLOOM_TARGET_BOS_TOKEN", "") or _DEFAULT_BOS),
    )
    # Same registry as the local path: raises for an unregistered model rather than
    # guessing a wrapper. Self-jail only here, so both wrappers are the target's.
    core._set_think_prefixes(target_model_id, target_model_id)
    return {"client": client,
            "target_no_think": core.think_prefix(target_model_id),
            "corrupt_no_think": core.think_prefix(target_model_id)}


def _jail_generate_api(handle: Dict, jail_runtime_cfg: Dict,
                       target_msgs_batch: List[List[Dict]], max_tokens: int,
                       temperature: float, no_think_target: bool) -> List[Dict]:
    """[jail engine=api_tilt] Drop-in for `_jail_generate_hf` over a hosted API.

    Returns one {"best_text", "best_ids", "best_token_probs"} per scenario, where
    best_token_probs are the UNMODIFIED-TARGET (temp=1) probabilities of the sampled
    tokens — the same plausibility metric the local engine reports. The elicited-only
    path additionally returns "best_token_probs_jail", the probabilities under the
    distribution the tokens were actually drawn from (free, and the natural input to
    the later top-k tilt approximation).
    """
    client: ApiTiltTarget = handle["client"]
    target_only = bool(jail_runtime_cfg.get("target_only"))
    b2 = float(jail_runtime_cfg.get("b2", 2.0))
    _b1 = jail_runtime_cfg.get("b1")
    b1 = float(_b1) if _b1 is not None else 1.0

    # ── refuse anything a text API cannot reproduce exactly ──
    if not target_only:
        floor = float(jail_runtime_cfg.get("target_floor", 0.0) or 0.0)
        if floor > 0.0:
            raise RuntimeError(
                f"api_tilt cannot apply jailbroken_output.target_floor={floor} — the floor masks "
                "on the TRUE TARGET distribution at every sampling step, which needs target "
                "logits while decoding. Set BLOOM_JAIL_FLOOR=0 (the elicited-only arm is "
                "defined without a target term anyway).")
        if float(jail_runtime_cfg.get("b3", 0.0) or 0.0) != 0.0:
            raise RuntimeError("api_tilt does not support jail negative steering (b3 != 0).")
        if float((jail_runtime_cfg.get("tokbias") or {}).get("lambda", 0.0) or 0.0) != 0.0:
            raise RuntimeError("api_tilt does not support the tokbias logit-bias baseline.")
        if not (b1 == 0.0 and b2 != 0.0):
            raise RuntimeError(
                f"api_tilt supports only the two mixing-free corners of the tilt: b1=1,b2=0 "
                f"(target only) and b1=0,b2!=0 (elicited only). Got b1={b1}, b2={b2}. A genuine "
                f"mix needs full-vocab logits from BOTH contexts at every step; approximating it "
                f"from top-k alternatives is a separate experiment, not a silent fallback.")

    NO_THINK = handle.get("target_no_think", "")
    NO_THINK_C = handle.get("corrupt_no_think", "")
    sys_prompt = jail_runtime_cfg.get("system_prompt", "")
    prefill = jail_runtime_cfg.get("prefill", "") or ""

    def _target_prefix(tm: List[Dict]) -> str:
        p = client.render(tm, add_generation_prompt=True)
        return p + NO_THINK if no_think_target else p

    def _jail_prefix(tm: List[Dict]) -> str:
        # Same construction as _jail_generate_hf: drop the target system prompt, prepend the
        # jail persona, always close the think block, then append the prefill. The prefill
        # conditions the jail distribution but is never sampled.
        conv = [m for m in tm if m.get("role") != "system"]
        j_msgs = ([{"role": "system", "content": sys_prompt}] + conv) if sys_prompt else conv
        return client.render(j_msgs, add_generation_prompt=True) + NO_THINK_C + prefill

    def _one(tm: List[Dict]) -> Dict:
        if target_only:
            # Drawn from the target itself -> its own generation logprobs are the on-policy
            # target probs. No second pass.
            g = client.generate(_target_prefix(tm), max_tokens, temperature)
            probs = [math.exp(l) * 100 for l in g["logprobs"]]
            return {"best_text": (g["text"] or "").strip(), "best_ids": g["ids"],
                    "best_token_probs": probs}
        # Elicited only: sample from the jail context, then score those exact tokens under
        # the TARGET context to get the plausibility metric.
        g = client.generate(_jail_prefix(tm), max_tokens, temperature)
        text, ids = g["text"] or "", g["ids"]
        if not ids:
            return {"best_text": "", "best_ids": [], "best_token_probs": []}
        t_lps = client.score_ids(_target_prefix(tm), ids)
        return {"best_text": text.strip(), "best_ids": ids,
                "best_token_probs": [math.exp(l) * 100 for l in t_lps],
                "best_token_probs_jail": [math.exp(l) * 100 for l in g["logprobs"]]}

    if len(target_msgs_batch) == 1:
        return [_one(target_msgs_batch[0])]
    with ThreadPoolExecutor(max_workers=min(len(target_msgs_batch), 16)) as ex:
        return list(ex.map(_one, target_msgs_batch))


__all__ = ["ApiTiltTarget", "load_api_target", "_jail_generate_api"]
