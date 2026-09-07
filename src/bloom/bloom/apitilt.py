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
import threading
import time
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

# Control markers that must never survive into a stored reply. gpt-oss keeps generating
# past its answer into a new harmony block (`<|end|><|start|>assistant<|channel|>...`);
# the decode stops at EOS (<|return|>) but not at <|end|>, so those tags reach the
# transcript and the template then REFUSES to re-render that turn:
#   "You have passed a message containing <|channel|> tags in the content field."
# Truncating at the first marker keeps the answer and drops the runaway continuation.
_STOP_MARKERS = ("<|end|>", "<|start|>", "<|channel|>", "<|message|>", "<|return|>",
                 "<|call|>", "<|constrain|>")


def _clip(text: str) -> str:
    """Cut a decoded reply at the first control marker, if any."""
    cut = min((text.find(m) for m in _STOP_MARKERS if m in text), default=-1)
    return (text[:cut] if cut >= 0 else text).strip()


_FINAL_OPEN = "<|channel|>final<|message|>"


def _finalize(res: "_TokenResolver", ids: List[int]) -> str:
    """The reply to store: for a harmony model, the FINAL channel only.

    _clip alone cannot do this. It searches the decoded text for control markers, but
    decode() strips them (skip_special_tokens=True), so it found nothing and the analysis
    channel reached the transcript with its tags erased -- turn 2 was then re-rendered as a
    final-channel message whose body was CoT plus several concatenated pseudo-turns.
    Reading the RAW decode instead lets the channel structure be recovered.

    Models without harmony channels (DeepSeek, GLM) have no _FINAL_OPEN in the raw decode
    and fall through to the previous behaviour unchanged.
    """
    raw = res.decode_raw(ids)
    i = raw.rfind(_FINAL_OPEN)
    if i < 0:
        return _clip(res.decode(ids))
    seg = raw[i + len(_FINAL_OPEN):]
    cut = min((seg.find(m) for m in _STOP_MARKERS if m in seg), default=-1)
    return (seg[:cut] if cut >= 0 else seg).strip()


class _TokenResolver:
    """Maps a top-k candidate STRING back to its token id.

    The provider returns alternatives as text only, but the running context is a list of
    token ids, so a chosen candidate has to be resolved before it can be appended. The
    vocabulary cannot be looked up directly: it stores byte-level keys (a leading space is
    `\u0120`, not " "), so `vocab[" right"]` misses. Decoding every id instead gives the
    same surface form the API reports, which matches exactly.

    Verified against 8767 real (string, id) pairs from the collected runs: 8722 exact, the
    rest the end-of-sequence token (handled by id), and all 4400 distinct top-5 candidate
    strings resolve.
    """

    def __init__(self, path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self._by_text: Dict[str, int] = {}
        # 36 strings (all U+FFFD byte-fallback variants) decode from more than one id.
        # Candidates arrive as TEXT, so a colliding string cannot be resolved -- keeping the
        # lowest id emitted a token the model never proposed (measured: '�' reported at
        # 82.55% resolved to id 97, true probability 1.04e-13). Dropped from the map instead,
        # so id_of returns None and the caller counts it in n_unres.
        _ambiguous: set = set()
        for i in range(self._tok.get_vocab_size()):
            s = self._tok.decode([i], skip_special_tokens=False)
            if s in self._by_text:
                _ambiguous.add(s)
            else:
                self._by_text[s] = i
        for s in _ambiguous:
            self._by_text.pop(s, None)
        self.n_ambiguous = len(_ambiguous)
        # A single eos id is not enough: several models end an assistant turn on a token
        # that is NOT their vocab-level EOS, and picking the wrong one is silent -- the loop
        # simply never stops and runs to max_tokens.
        #   gpt-oss  eos resolved to <|endoftext|> (199999), which a chat turn never emits;
        #            the real enders are <|return|> and <|call|>. Measured consequence: 45/45
        #            replies at the 250-token cap and 44/45 carrying the model's analysis
        #            channel into the stored turn, against 0/45 for the same cell's vanilla.
        #   GLM      eos resolves to <|endoftext|> but turns end on <|user|>/<|observation|>.
        # <|end|> is deliberately NOT a stop for gpt-oss: it closes the analysis channel
        # mid-reply, and gpt_oss_chat.jinja says so outright ("<|return|> indicates the end
        # of generation, but <|end|> does not"). Stopping there would keep only the CoT.
        _v = self._tok.get_vocab()
        _stop_names = ("<｜end▁of▁sentence｜>", "<|endoftext|>", "<|return|>", "<|call|>",
                       "<|im_end|>", "<|eot_id|>", "</s>", "<|user|>", "<|observation|>")
        self.stop_ids = {_v[k] for k in _stop_names if k in _v}
        self.eos_id = next((_v[k] for k in ("<｜end▁of▁sentence｜>",
                                            "<|endoftext|>", "<|return|>", "<|im_end|>", "</s>")
                            if k in _v), 1)
        if not self.stop_ids:
            self.stop_ids = {self.eos_id}

    def id_of(self, text: str) -> Optional[int]:
        return self._by_text.get(text)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)

    def decode_raw(self, ids: List[int]) -> str:
        """Decode KEEPING control tokens, so channel structure can be read."""
        return self._tok.decode(list(ids), skip_special_tokens=False)


# One resolver per tokenizer path, per process.
_RESOLVER: Dict[str, _TokenResolver] = {}


def _resolver() -> _TokenResolver:
    """Lazily build the reverse map (~129k decodes, a few seconds) once per process."""
    path = (os.environ.get("BLOOM_TARGET_TOKENIZER", "") or "").strip()
    if not path:
        path = str(Path.home() / ".cache" / "bloom" / "dsv4_tokenizer.json")
    if not Path(path).exists():
        raise RuntimeError(
            f"api_tilt rule=overlap needs the target's tokenizer.json to turn top-k candidate "
            f"strings back into token ids, and none is at {path}. Fetch it once:\n"
            f"  curl -L -o {path} https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731"
            f"/resolve/main/tokenizer.json\n"
            f"or point BLOOM_TARGET_TOKENIZER at one.")
    if path not in _RESOLVER:
        _RESOLVER[path] = _TokenResolver(path)
    return _RESOLVER[path]


class ApiTiltTarget:
    """One hosted model, addressed through /completions for both generation and scoring."""

    def __init__(self, model: str, provider: str = "fireworks",
                 template_path: str = "", bos_token: str = _DEFAULT_BOS,
                 timeout: float = 25.0, max_retries: int = 6):
        if provider not in _PROVIDERS:
            raise RuntimeError(
                f"BLOOM_TARGET_API={provider!r} unknown; known providers: {sorted(_PROVIDERS)}")
        p = _PROVIDERS[provider]
        key = (os.environ.get(p["key_env"], "") or "").strip()
        if not key:
            raise RuntimeError(
                f"{p['key_env']} is not set — the api_tilt engine needs it to reach {provider}.")
        self.model, self.provider, self.base = model, provider, p["base"]
        # 25s, measured at the prompt sizes and concurrency this engine actually uses
        # (200-2000 token contexts, 30 in flight). Median is 1.3-4.5s and p90 is 5-15s
        # regardless of prompt length -- length is NOT the driver, a fat tail of 30-60s
        # cold starts is -- so 25s sits above p90 while still abandoning the tail.
        # 8s was measured on 20-token prompts and killed every run: it fell below p90, so
        # calls failed constantly.
        self.timeout = float(os.environ.get("BLOOM_API_TIMEOUT", "") or timeout)
        self.max_retries = max_retries
        self.n_retries = 0
        # Cloudflare in front of some providers rejects the default urllib User-Agent.
        self._hdr = {"User-Agent": "curl/8.4.0", "Content-Type": "application/json",
                     "Accept": "application/json", "Authorization": f"Bearer {key}"}
        if not template_path:
            raise RuntimeError(
                "api_tilt needs BLOOM_TARGET_CHAT_TEMPLATE — prompts are rendered here, not by "
                "a tokenizer, so the template is the only thing pinning the API's tokenization "
                "to the local run's.")
        # Match transformers' _compile_jinja_template exactly: loopcontrols for templates
        # using {% break %} (GLM-5.3 does), plus trim_blocks/lstrip_blocks and the tojson
        # filter it installs. Anything else risks rendering a prompt the server tokenizes
        # differently from what we think we sent.
        import jinja2.ext
        from jinja2 import Environment
        _env = Environment(trim_blocks=True, lstrip_blocks=True,
                           extensions=[jinja2.ext.loopcontrols])
        _env.filters["tojson"] = lambda o, **kw: json.dumps(o, **kw)
        _env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
        # gpt-oss's harmony template calls strftime_now to stamp a date into the system
        # block. NOTE this makes its prompts date-dependent: runs on different days are not
        # byte-identical, unlike DeepSeek's and GLM's.
        import datetime as _dt
        _env.globals["strftime_now"] = lambda f: _dt.datetime.now().strftime(f)
        self._tpl = _env.from_string(Path(template_path).read_text(encoding="utf-8"))
        self._bos = bos_token
        # One pooled, keep-alive session PER THREAD. A fresh TCP+TLS connection per call
        # is not merely slower -- measured against this endpoint it failed 3 of 12 times
        # (hung sockets, seen as CLOSE_WAIT), while a pooled session failed 0 of 12 at the
        # same latency. requests.Session is not documented thread-safe, and the driven
        # decode runs ~30 calls in flight, so each thread gets its own.
        self._tl = threading.local()
        # Fireworks' priority serving path. Standard is serverless and a request routed to
        # a cold replica stalls tens of seconds in pure time-to-first-token; priority
        # suppresses that (measured: worst TTFT 5.1s against 26.8s on standard at the same
        # load, and 0.26-1.5s vs 4.9-6.3s serial). It bills at ~1.25-1.5x, so it is opt-in.
        # Verified to work on /completions BOTH with and without echo.
        self.service_tier = (os.environ.get("BLOOM_API_SERVICE_TIER", "") or "").strip() or None
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
    def _session(self):
        s = getattr(self._tl, "s", None)
        if s is None:
            import requests
            s = requests.Session()
            s.headers.update(self._hdr)
            # max_retries=0: the loop below owns retry policy, so urllib3 must not also
            # retry underneath it (that would multiply the wait before we ever see a failure).
            s.mount("https://", requests.adapters.HTTPAdapter(
                pool_connections=4, pool_maxsize=8, max_retries=0))
            self._tl.s = s
        return s

    def _post(self, body: Dict, affinity: Optional[str] = None) -> Dict:
        if self.service_tier:
            body = dict(body, service_tier=self.service_tier)
        # Fireworks' prompt cache lives in ONE replica, so a sequence of incrementally
        # growing prompts only reuses its prefix if every call lands on the same one.
        # Without this header the driven decode below scatters across replicas and
        # silently pays full prefill every step (verified: cached_tokens goes 899 -> 0).
        last = ""
        for attempt in range(self.max_retries):
            _throttled = False      # True only for 429/5xx, which want real backoff
            # Affinity pins this request to ONE replica so the prefix cache hits. That is
            # what we want on the first attempt and exactly what we do NOT want after a
            # failure: a cold replica would receive every retry and time out identically,
            # which is how a single bad replica killed whole runs. Retries drop the header
            # so they can route somewhere else, trading a cache miss for a live replica.
            hdr = {"x-session-affinity": affinity} if (affinity and attempt == 0) else None
            try:
                r = self._session().post(self.base + "/completions", json=body,
                                         headers=hdr, timeout=self.timeout)
                if r.status_code == 200:
                    self.n_calls += 1
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                # 4xx other than rate-limit is a request bug — retrying just burns time.
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    raise RuntimeError(f"api_tilt request rejected — {last}")
                _throttled = True
            except RuntimeError:
                raise
            except Exception as e:                      # timeout / connection reset
                last = f"{type(e).__name__}: {e}"
                # A broken pooled connection must not be reused for the retry.
                self._tl.s = None
            # Retries were previously silent, which made a hung socket look exactly like
            # ordinary slowness in the run log. Say so.
            self.n_retries += 1
            print(f"  [api_tilt] retry {attempt + 1}/{self.max_retries} after {last}", flush=True)
            # Back off only for the failures backoff is FOR. A timeout here means the
            # request hit a cold replica, so the useful response is to try again promptly
            # on a fresh connection and hope to route elsewhere -- sleeping just adds the
            # delay back that the short timeout was meant to avoid. Rate limiting and 5xx
            # are the cases that genuinely want exponential backoff.
            if _throttled:
                time.sleep(min(30.0, 2.0 ** attempt) * (0.5 + random.random()))
            else:
                time.sleep(0.15 + 0.35 * random.random())
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

    def floored_target_sample(self, ids: List[int], top_k: int, temperature: float,
                              floor_pct: float, p_max: float,
                              affinity: Optional[str] = None) -> Dict:
        """A draw from the TARGET whose probability cannot fall below `floor_pct` (percent).

        min_p is relative (p >= min_p * p_max), so the absolute floor is expressed
        per-position as floor/p_max -- p_max is already known from the target's own top-k, so
        this costs one call and no extra scoring.
        """
        mp = (floor_pct / 100.0) / p_max if p_max > 0 else None
        if mp is not None and mp > 1.0:
            # The floor exceeds p_max: NO token can clear it. Falling back to a free draw
            # here would emit anything at all, so take the argmax instead -- the closest the
            # position can get to the floor.
            return self.next_topk(ids, top_k, 0.0, affinity)
        if mp is None or mp <= 0.0:
            mp = None
        return self.next_topk(ids, top_k, temperature, affinity, min_p=mp)

    def cand_logprob(self, ctx_ids: List[int], cand_id: int) -> float:
        """Teacher-forced logprob of `cand_id` as the NEXT token after `ctx_ids`.

        Prices a candidate outside the target's own top-k. Ids, never text, so the position is
        exact. One max_tokens=0 call, only on empty-overlap positions (~1% of tokens).
        """
        lp = self._echo(list(ctx_ids) + [int(cand_id)])
        ids = list(lp.get("token_ids") or [])
        lps = list(lp.get("token_logprobs") or [])
        if len(ids) != len(ctx_ids) + 1 or ids[-1] != int(cand_id):
            raise RuntimeError(
                f"api_tilt cand_logprob: provider did not echo the candidate id back "
                f"(sent {len(ctx_ids)}+1, got {len(ids)}).")
        return float(lps[-1])

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

    def next_topk(self, ids: List[int], top_k: int = 5, temperature: float = 1.0,
                  affinity: Optional[str] = None,
                  exclude_ids: Optional[List[int]] = None,
                  min_p: Optional[float] = None) -> Dict:
        """One decode step: the top-`top_k` candidates for the position after `ids`.

        Asks for a single token so the response carries `top_logprobs[0]` — the
        distribution at that position — plus the token the provider itself sampled from
        the full (untruncated) distribution, which the empty-overlap fallback uses
        directly. `top_p`/`top_k` are pinned wide open so that sample is a genuine draw
        from the whole distribution, not a truncated one.

        `exclude_ids` suppresses tokens via logit_bias -> sampling WITHOUT replacement across
        calls. Needed because independent redraws at a peaked position return the same token
        nearly every time, which is exactly where a resample loop must find an alternative.

        Returns {"top": [(token_string, logprob) x top_k], "sampled_id", "sampled_str",
        "sampled_lp", "cached"}.
        """
        _body = {"model": self.model, "prompt": list(ids), "max_tokens": 1,
                 "temperature": float(temperature), "top_p": 1.0, "top_k": 0,
                 "logprobs": int(top_k), "echo": False}
        if exclude_ids:
            _body["logit_bias"] = {str(int(i)): -100 for i in exclude_ids}
        if min_p is not None:
            # Relative, not absolute: admits tokens with p >= min_p * p_max. Verified
            # enforced on Fireworks (min_p=0.9 collapsed 14 draws to one token where the
            # unrestricted prompt gave nine). The absolute form, epsilon_cutoff, is rejected.
            _body["min_p"] = float(min_p)
        r = self._post(_body, affinity=affinity)
        ch = r["choices"][0]
        lp = ch.get("logprobs") or {}
        u = r.get("usage") or {}
        self.n_prompt_tokens += int(u.get("prompt_tokens") or 0)
        self.n_gen_tokens += int(u.get("completion_tokens") or 0)
        top = sorted(((lp.get("top_logprobs") or [{}])[0] or {}).items(), key=lambda kv: -kv[1])
        sid = (lp.get("token_ids") or [None])[0]
        return {"top": top,
                "sampled_id": (int(sid) if sid is not None else None),
                "sampled_str": (lp.get("tokens") or [""])[0],
                "sampled_lp": float((lp.get("token_logprobs") or [0.0])[0] or 0.0),
                "cached": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)}

    def stats(self) -> str:
        return (f"{self.n_calls} calls, {self.n_retries} retries, "
                f"{self.n_prompt_tokens} prompt tok, {self.n_gen_tokens} generated tok"
                + (f", tier={self.service_tier}" if self.service_tier else ""))


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


def _record_cost(client, tag: str, extra: Optional[Dict] = None) -> None:
    """Append one cost record to <BLOOM_RUNS_ROOT>/<BLOOM_FOLDER>/api_cost.jsonl.

    Recorded, never printed. Cumulative counters, so diffing two lines gives one batch. Calls
    and tokens sit alongside wall time because wall time is not comparable across runs -- a
    degraded endpoint moved the same arm from ~100 to ~500 s/1k tokens. Failures swallowed.
    """
    try:
        root = os.environ.get("BLOOM_RUNS_ROOT", "") or ""
        folder = os.environ.get("BLOOM_FOLDER", "") or ""
        if not folder:
            return
        d = os.path.join(root, folder)
        os.makedirs(d, exist_ok=True)
        rec = {"t": time.time(), "tag": tag,
               "calls": getattr(client, "n_calls", None),
               "retries": getattr(client, "n_retries", None),
               "prompt_tok": getattr(client, "n_prompt_tokens", None),
               "gen_tok": getattr(client, "n_gen_tokens", None)}
        if extra:
            rec.update(extra)
        with open(os.path.join(d, "api_cost.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + chr(10))
    except Exception:
        pass


def _driven_overlap(handle: Dict, jail_runtime_cfg: Dict,
                    target_msgs_batch: List[List[Dict]], max_tokens: int,
                    temperature: float, no_think_target: bool) -> List[Dict]:
    """[api_tilt rule=overlap] Token-by-token decode that combines the two contexts.

    Both contexts are advanced in lockstep over the SAME emitted tokens, and at each
    position:

      * ask each context for its top-5 next-token candidates (2 calls, issued together);
      * if the two top-5 sets overlap, emit the overlap member the ELICITED context ranks
        highest -- the behaviour the jail context wants, restricted to something the target
        also considers a live candidate;
      * if the overlap is empty, emit the token the TARGET context sampled on its own. That
        is an ordinary draw from the full target distribution, not a top-5 pick, and it
        costs nothing extra because the target call already sampled one.

    Reported plausibility stays what it is everywhere else in this pipeline: the unmodified
    TARGET probability of the emitted token. It is always known -- from the target's top-5
    when the token came from the overlap, and from the target call's own token logprob when
    it came from the fallback.

    Both calls per step are the SAME shape and path, so the two contexts are numerically
    comparable to each other; they are on the generation path, not the echo path, which
    differs by ~2 pp (see the module docstring's note).
    """
    client: ApiTiltTarget = handle["client"]
    res = _resolver()
    NO_THINK = handle.get("target_no_think", "")
    NO_THINK_C = handle.get("corrupt_no_think", "")
    sys_prompt = jail_runtime_cfg.get("system_prompt", "")
    prefill = jail_runtime_cfg.get("prefill", "") or ""
    top_k = int(jail_runtime_cfg.get("api_top_k", 5) or 5)
    # What to emit when stage 1 cannot fill a position. jail_resample draws from the
    # ELICITED side and needs the post-hoc rescore below, since an elicited token is usually
    # outside the target top-k and its logprob is unknown at decode time; target_sample takes
    # the target's own draw, which costs nothing because that call already sampled one.
    fb_mode = str(jail_runtime_cfg.get("api_fallback", "jail_descend") or "jail_descend")
    if fb_mode not in ("jail_resample", "jail_descend", "target_sample"):
        raise RuntimeError(f"api_jailbroken_output.fallback={fb_mode!r} unknown "
                           f"(jail_resample | jail_descend | target_sample)")
    # MIXTURE weights, in PROBABILITY space over the UNION of the two top-k sets:
    # score = b1*p_target + b2*p_elicited, a side that did not propose a token contributing 0.
    # Unlike a product over the intersection it does not need both contexts to like a token,
    # and because the union always contains the target's own top-1, b2=0 is greedy vanilla.
    _ob1 = jail_runtime_cfg.get("b1")
    ob1 = float(_ob1) if _ob1 is not None else 1.0
    ob2 = float(jail_runtime_cfg.get("b2", 1.0))
    # FLOOR (percent): one value governing BOTH stages. In stage 1 it rejects candidates the
    # target prices below it -- target-side members directly, and elicited-only members via
    # the free bound t(x) <= min_T(t), falling back to one cand_logprob call on the winner
    # only when that bound cannot settle it. In stage 2 it is the bar an elicited draw must
    # clear to be accepted. It exists because the union scores a token the target did not
    # propose as b1*0 + b2*p_e, so it cannot tell a token the target rates 1e-3 from one it
    # rates 1e-12: measured, the unfloored arm's worst tokens were elicited-only picks at
    # 1e-10..1e-08 % with the elicited context 56-93% confident.
    floor = float(jail_runtime_cfg.get("api_floor", 0.0) or 0.0)
    # What to do when the stage-1 winner is an elicited-only token priced below the floor.
    # "repick" drops it and re-argmaxes, staying in stage 1 with the next-best candidate.
    # "stage2" hands the position to the elicited resample instead.
    floor_action = str(jail_runtime_cfg.get("api_floor_action", "stage2") or "stage2")
    if floor_action not in ("repick", "stage2"):
        raise RuntimeError(f"api_jailbroken_output.floor_action={floor_action!r} unknown "
                           f"(repick | stage2)")
    # STAGE-2 TRIGGER, on q = the share of the ELICITED context's own top-k mass sitting on
    # tokens the overlap cannot deliver. Keying on the elicited side is deliberate: the
    # question is whether what that context wants is reachable, not whether the target is
    # uncertain. q = 1 exactly when the two top-k sets are disjoint.
    #   threshold -- escalate iff q >= stage2_theta. Deterministic, so the trigger adds no
    #                run-to-run variance. theta=1 fires only on disjoint sets; theta -> 0
    #                approaches always-escalate.
    #   never     -- stage 2 off entirely. The mixture scores the union and takes its argmax,
    #                which is never empty because the target's own top-k is in it.
    stage2_mode = str(jail_runtime_cfg.get("api_stage2", "threshold") or "threshold")
    if stage2_mode not in ("threshold", "never"):
        raise RuntimeError(f"api_jailbroken_output.stage2={stage2_mode!r} unknown "
                           f"(threshold | never)")
    stage2_theta = float(jail_runtime_cfg.get("api_stage2_theta", 0.95) or 0.95)
    # HOW DISAGREEMENT IS MEASURED. Every schedule alpha(q) is monotone in q, so the shape
    # only sets how many positions are steered and how hard -- it cannot change WHICH ones.
    # Only the metric reorders positions, so it is the one structural lever here.
    #   elicited_outside -- share of the elicited top-k mass on tokens the target did not
    #                       propose. Blind to rank disagreement INSIDE the overlap: target
    #                       {A .90, B .05} against elicited {A .05, B .90} scores 0.
    #   tv               -- total variation between the two top-k distributions, each
    #                       renormalised over its own top-k. Strict generalisation: still 1
    #                       when the sets are disjoint, but also fires on the rank flip above.
    #   margin           -- p_e(elicited top-1) - p_e(target top-1): how much the ELICITED
    #                       context gains by intervening, rather than how far apart the two
    #                       distributions are. Near 0 where it is nearly indifferent (steering
    #                       there spends plausibility for nothing) and near 1 where it wants
    #                       something the target would never pick.
    q_metric = str(jail_runtime_cfg.get("api_q_metric", "elicited_outside") or "elicited_outside")
    if q_metric not in ("elicited_outside", "tv", "margin"):
        raise RuntimeError(f"api_jailbroken_output.q_metric={q_metric!r} unknown "
                           f"(elicited_outside | tv | margin)")
    # The last two stochastic paths in the decode: an unresolvable surface form, and the
    # revert when nothing the elicited side offers clears the floor. Both otherwise take a
    # DRAW from the target (the latter min_p-constrained). With det_fallback they take the
    # target's top-1 instead, which makes the whole decode a deterministic function of the
    # two contexts' top-k -- the only remaining variation is then the hosted model's own
    # non-determinism in those top-k values.
    det_fallback = bool(jail_runtime_cfg.get("api_det_fallback", True))
    # Duty cycle on the steering: every Nth generated position ignores both stages and
    # emits the TARGET's top-1, so the reply alternates between steered and greedy tokens.
    # 0 disables. Greedy vanilla runs at 83.06% arithmetic against the steered arms' 65-71%,
    # so interleaving trades behaviour for plausibility at a rate set by N -- a coarser
    # version of spending a plausibility budget, with the spend spread uniformly.
    target_every = int(jail_runtime_cfg.get("api_target_every", 0) or 0)
    # Emit a DRAW from the mixture scores rather than their argmax, proportional to
    # score**(1/sample_temp). 0 keeps the argmax. Sharpening matters: an unsharpened draw
    # (T=1) is a coin flip wherever the two contexts disagree and previously tripled the
    # share of tokens the target rates under 10%, while T<=0.2 recovered most of that. The
    # point of sampling is round-to-round diversity for pools and post-run selection, which
    # a deterministic decode cannot provide.
    sample_temp = float(jail_runtime_cfg.get("api_sample_temp", 0.05) or 0.05)
    # ADAPTIVE WEIGHT (adaptive=True). The union score is a convex mixture
    #     score(x) = alpha*p_target(x) + (1-alpha)*p_elicited(x)
    # and alpha is set per position from the measured disagreement q:
    #     alpha(q) = alpha0 * (1 - q**k)
    # Only the RATIO of the two weights affects an argmax, so this is the b1/b2 pair with
    # its redundant degree of freedom removed: alpha0 = b1/(b1+b2), and b2=1 <-> alpha0=0.5.
    # At q=1 (the two top-k sets disjoint) alpha=0, i.e. pure elicited control -- which is
    # exactly what the stage-2 branch does today, so the branch is not needed. k sets how
    # sharply control transfers: large k holds alpha at alpha0 until q approaches 1 (the
    # current threshold behaviour), small k hands over early and in proportion.
    adaptive = bool(jail_runtime_cfg.get("api_adaptive", True))
    alpha0 = float(jail_runtime_cfg.get("api_alpha0", 0.6))
    alpha_k = float(jail_runtime_cfg.get("api_alpha_k", 10.0) or 10.0)
    if adaptive and not (0.0 <= alpha0 <= 1.0):
        raise RuntimeError(f"api_jailbroken_output.alpha0={alpha0!r} must be in [0, 1]")
    if adaptive and alpha_k <= 0.0:
        raise RuntimeError(f"api_jailbroken_output.alpha_k={alpha_k!r} must be > 0")
    # jail_resample only: how many draws from the elicited distribution to try before giving
    # up and keeping the most target-plausible of them.
    fb_tries = int(jail_runtime_cfg.get("api_fb_tries", 5) or 5)

    def _targ_argmax(tmap, jmap, res):
        """(id, target_lp, elicited_lp) for the target's top-1, or None if unresolvable."""
        if not tmap:
            return None
        _am = max(tmap, key=tmap.get)
        _amid = res.id_of(_am)
        return None if _amid is None else (_amid, tmap[_am], jmap.get(_am))

    def _one(job):
        idx, tm = job
        aff = f"tilt-{os.getpid()}-{idx}"
        t_prefix = client.render(tm, add_generation_prompt=True) + (NO_THINK if no_think_target else "")
        conv = [m for m in tm if m.get("role") != "system"]
        j_msgs = ([{"role": "system", "content": sys_prompt}] + conv) if sys_prompt else conv
        j_prefix = client.render(j_msgs, add_generation_prompt=True) + NO_THINK_C + prefill
        # Prefix ids come from the API's own tokenizer, so the starting context is exact;
        # the local tokenizer is only ever used to resolve ONE candidate string at a time.
        t_ids = client.prefix_ids(t_prefix)
        j_ids = client.prefix_ids(j_prefix)

        gen, t_lps, j_lps, n_fallback, n_unres, n_floored = [], [], [], 0, 0, 0
        n_resamples = 0
        n_shortcut = 0
        n_empty = 0
        alpha_sum = 0.0      # mean adaptive alpha, for the run summary
        n_greedy = 0         # positions handed to the target argmax by target_every
        n_mixdrop = 0        # candidates removed by the stage-1 floor
        n_mixcalls = 0       # cand_logprob calls the free bound could not avoid
        q_sum = 0.0
        truncated = ""
        with ThreadPoolExecutor(max_workers=2) as ex:
            for _ in range(int(max_tokens)):
                ft = ex.submit(client.next_topk, t_ids, top_k, temperature, aff + "-t")
                fj = ex.submit(client.next_topk, j_ids, top_k, temperature, aff + "-j")
                try:
                    tr, jr = ft.result(), fj.result()
                except RuntimeError as e:
                    # A call that exhausts its retries used to propagate out of the thread
                    # pool and abort the ENTIRE round -- every scenario, every completed
                    # turn, discarded because of one bad request. End this scenario's reply
                    # here instead: a short transcript is worth incomparably more than
                    # losing the other fourteen and the work already done.
                    truncated = str(e)
                    break
                tmap = dict(tr["top"])
                if target_every > 0 and (len(gen) + 1) % target_every == 0 and tmap:
                    # Greedy position: the target's own top-1, no steering at all.
                    _am = max(tmap, key=tmap.get)
                    _amid = res.id_of(_am)
                    if _amid is not None:
                        n_greedy += 1
                        if _amid in res.stop_ids:
                            break
                        gen.append(_amid)
                        t_lps.append(tmap[_am])
                        j_lps.append(dict(jr["top"]).get(_am, float("nan")))
                        t_ids = t_ids + [_amid]
                        j_ids = j_ids + [_amid]
                        continue
                overlap = [(s, lp) for s, lp in jr["top"] if s in tmap]
                if not overlap:
                    n_empty += 1
                # Disagreement share of the elicited top-k; q = 1 exactly when the two
                # top-k sets are disjoint.
                _ov = {x[0] for x in overlap}
                _te = sum(math.exp(lp) for _, lp in jr["top"]) or 1.0
                if q_metric == "elicited_outside":
                    _q = sum(math.exp(lp) for _s, lp in jr["top"] if _s not in _ov) / _te
                elif q_metric == "tv":
                    # Each side renormalised over its own top-k so both are proper
                    # distributions; TV is then in [0,1] and reaches 1 on disjoint sets.
                    _tt = sum(math.exp(lp) for _, lp in tr["top"]) or 1.0
                    _pt = {_s: math.exp(lp) / _tt for _s, lp in tr["top"]}
                    _pe = {_s: math.exp(lp) / _te for _s, lp in jr["top"]}
                    _q = 0.5 * sum(abs(_pt.get(_s, 0.0) - _pe.get(_s, 0.0))
                                   for _s in set(_pt) | set(_pe))
                else:   # margin
                    # What the ELICITED context gains by getting its way at this position.
                    _pe = {_s: math.exp(lp) / _te for _s, lp in jr["top"]}
                    _etop = max(_pe, key=_pe.get) if _pe else None
                    _ttop = tr["top"][0][0] if tr["top"] else None
                    _q = max(0.0, (_pe.get(_etop, 0.0) - _pe.get(_ttop, 0.0)) if _etop else 0.0)
                _q = min(1.0, max(0.0, _q))
                q_sum += _q
                # Under the adaptive weight, q=1 already drives alpha to 0 and the argmax
                # becomes the elicited top-1 among floor-passing candidates -- which is what
                # stage 2 emits -- so no separate branch fires.
                _stage2 = False if (adaptive or stage2_mode == "never") else (_q >= stage2_theta)
                # Score over the UNION, in probability space.
                _union = {}
                for _t, _lp in tr["top"]:
                    _union[_t] = [math.exp(_lp), 0.0]
                for _t, _lp in jr["top"]:
                    _union.setdefault(_t, [0.0, 0.0])[1] = math.exp(_lp)
                if adaptive:
                    # Clamped away from exactly 0 so the target's own ordering still breaks
                    # ties among candidates the elicited context never proposed -- without
                    # it, at q=1 every target-only candidate scores 0 and the pick is
                    # arbitrary rather than the target's top-1.
                    _alpha = max(alpha0 * (1.0 - _q ** alpha_k), 1e-9)
                    alpha_sum += _alpha
                    _mix = [(t, _alpha * xy[0] + (1.0 - _alpha) * xy[1])
                            for t, xy in _union.items()]
                else:
                    _mix = [(t, ob1 * xy[0] + ob2 * xy[1]) for t, xy in _union.items()]
                if floor > 0.0:
                    _tmin = min(math.exp(v) for v in tmap.values()) * 100.0 if tmap else 0.0
                    # Free rejection of every elicited-only candidate when the target's own
                    # k-th best already fails the floor: nothing outside the top-k can beat it.
                    _drop_all_e = _tmin < floor
                    _keep = []
                    for _t, _w in _mix:
                        if _t in tmap:
                            if math.exp(tmap[_t]) * 100.0 < floor:
                                n_mixdrop += 1
                                continue
                            _keep.append((_t, _w))
                        elif not _drop_all_e:
                            _keep.append((_t, _w))     # unknown t(x); verified at pick time
                        else:
                            n_mixdrop += 1
                    _mix = _keep
                # Resolve the mixture argmax BEFORE the dispatch, so that a candidate list
                # emptied by the floor routes to stage 2 rather than falling through with a
                # stale pick.
                def _pick_from(cands):
                    """Argmax, or a sharpened draw when sample_temp > 0."""
                    if not cands:
                        return None
                    if sample_temp <= 0.0:
                        return max(cands, key=lambda x: x[1])[0]
                    # Renormalise to the largest weight before the power so a small
                    # temperature cannot underflow every candidate to zero.
                    _wmax = max(w for _, w in cands) or 1.0
                    _e = 1.0 / sample_temp
                    _d = [(t, (w / _wmax) ** _e) for t, w in cands]
                    _tot = sum(w for _, w in _d) or 1.0
                    _r, _acc = random.random() * _tot, 0.0
                    for _t, _w in _d:
                        _acc += _w
                        if _r <= _acc:
                            return _t
                    return _d[-1][0]

                _mixpick = None
                if _mix:
                    _mixpick = _pick_from(_mix)
                    # Only reachable when the free bound could not settle it: price the
                    # winner, then drop it or escalate if it fails.
                    while floor > 0.0 and _mixpick is not None and _mixpick not in tmap:
                        _pid = res.id_of(_mixpick)
                        _ok = False
                        if _pid is not None:
                            try:
                                n_mixcalls += 1
                                _ok = math.exp(client.cand_logprob(t_ids, _pid)) * 100.0 >= floor
                            except RuntimeError as e:
                                truncated = str(e)
                                break
                        if _ok:
                            break
                        n_mixdrop += 1
                        if floor_action == "stage2":
                            # Escalate this position instead of settling for second best.
                            _stage2 = True
                            _mixpick = None
                            break
                        _mix = [x for x in _mix if x[0] != _mixpick]
                        _mixpick = _pick_from(_mix)
                if truncated:
                    break
                if ((overlap or stage2_mode == "never" or adaptive)
                        and not _stage2 and _mixpick is not None):
                    # The union is scored everywhere, but stage 1 only OWNS a position when
                    # the two top-k sets actually intersect -- unless stage 2 is off, or the
                    # adaptive weight is in charge, in which case q=1 drives alpha to 0 and
                    # the union argmax IS the elicited top-1, so a disjoint
                    # position routes to the fallback below, which is where the behaviour
                    # that neither context agrees on has to come from.
                    pick = _mixpick
                    tid = res.id_of(pick)
                    if tid is None:
                        n_unres += 1
                        _ta = _targ_argmax(tmap, dict(jr["top"]), res) if det_fallback else None
                        if _ta is not None:
                            tid, t_lp, j_lp = _ta
                        else:
                            tid, t_lp = tr["sampled_id"], tr["sampled_lp"]
                            j_lp = dict(jr["top"]).get(tr["sampled_str"])
                    else:
                        t_lp = tmap.get(pick)          # None if outside the target top-k
                        j_lp = dict(jr["top"]).get(pick)
                else:
                    n_fallback += 1
                    jmap = dict(jr["top"])
                    tid = None
                    _forced_t_lp = None
                    _forced_j_lp = None
                    if fb_mode == "jail_descend":
                        # Walk the ELICITED top-k in its own rank order and emit the first
                        # candidate the target prices at or above the floor -- argmax, then
                        # second, and so on. Deterministic, so unlike the resample the same
                        # position resolves identically every run and the trigger is the only
                        # stochastic element left in the decode. Cost is at most k prices
                        # rather than up to fb_tries draws plus a price each.
                        _tmin = (min(math.exp(v) for v in tmap.values()) * 100.0) if tmap else 0.0
                        for _s, _slp in jr["top"]:
                            _cid = res.id_of(_s)
                            if _cid is None:
                                continue
                            if _s in tmap:
                                _tlp = tmap[_s]          # already priced, no call
                            elif floor > 0.0 and _tmin < floor:
                                continue                 # free bound: cannot clear the floor
                            else:
                                n_resamples += 1         # one cand_logprob probe
                                _tlp = client.cand_logprob(t_ids, _cid)
                            if floor <= 0.0 or math.exp(_tlp) * 100.0 >= floor:
                                tid, _forced_t_lp, _forced_j_lp = _cid, _tlp, _slp
                                break
                        if tid is None:
                            # The whole elicited top-k is below the floor; fall back to a
                            # floored target draw exactly as the resample does.
                            n_floored += 1
                            _ta = _targ_argmax(tmap, jmap, res) if det_fallback else None
                            if _ta is not None:
                                tid, _forced_t_lp, _forced_j_lp = _ta
                            else:
                                _fs = client.floored_target_sample(
                                    t_ids, top_k, temperature, floor,
                                    math.exp(max(tmap.values())) if tmap else 0.0, aff + "-t")
                                tid = _fs["sampled_id"]
                                _forced_t_lp = tmap.get(_fs["sampled_str"], float("nan"))
                                _forced_j_lp = jmap.get(_fs["sampled_str"])
                    elif (fb_mode == "jail_resample" and floor > 0.0 and tmap
                            and sum(1 for _lp in tmap.values()
                                    if math.exp(_lp) * 100.0 >= floor) <= 1):
                        # The target's top-k is sorted, so if at most one member clears the
                        # floor then at most one token in the WHOLE vocabulary does -- every
                        # token outside the top-k is below its smallest member. Resampling
                        # can only ever find that one token, and it is the argmax. Take it
                        # directly; where none clear, the argmax is still the closest.
                        n_shortcut += 1
                        _am = max(tmap, key=tmap.get)
                        _amid = res.id_of(_am)
                        if _amid is None:
                            n_unres += 1
                            tid, t_lp = tr["sampled_id"], tr["sampled_lp"]
                            j_lp = jmap.get(tr["sampled_str"])
                        else:
                            tid, t_lp, j_lp = _amid, tmap[_am], jmap.get(_am)
                    elif fb_mode == "jail_resample":
                        # Accept the first elicited draw clearing the floor, else the best of
                        # `fb_tries`. It only intervenes when a draw is actually bad, so the
                        # elicited preference survives where it is already acceptable.
                        # Stochastic, so rounds differ and pools stay diverse.
                        _tries = []
                        for _k in range(fb_tries):
                            if _k == 0:
                                _sid, _slp = jr["sampled_id"], jr["sampled_lp"]
                            else:
                                # Without replacement: a peaked position returns the same
                                # token on every redraw, so the loop would do nothing.
                                _jr2 = client.next_topk(
                                    j_ids, top_k, temperature, aff + "-j",
                                    exclude_ids=[x[1] for x in _tries])
                                # _slp is under the BIASED distribution, so it overstates the
                                # true elicited probability. Not corrected: nothing reported
                                # depends on it (all target-side), and freeselect's margin
                                # moves ~0.02 nats. Not worth a call per resample.
                                _sid, _slp = _jr2["sampled_id"], _jr2["sampled_lp"]
                            if _sid is None:
                                continue
                            _tlp = client.cand_logprob(t_ids, _sid)
                            _tries.append((_tlp, _sid, _slp))
                            if math.exp(_tlp) * 100.0 >= floor:
                                break
                        if _tries:
                            _bt, _bid, _blp = max(_tries, key=lambda x: x[0])
                            n_resamples += len(_tries) - 1
                            if math.exp(_bt) * 100.0 >= floor:
                                tid, _forced_t_lp, _forced_j_lp = _bid, _bt, _blp
                            else:
                                # Every draw failed. Keeping the best of them still emits an
                                # impossible token -- measured 1.16e-08 where the elicited
                                # context was 100% certain, so sampling without replacement
                                # only walked further into its tail. Give up on the elicited
                                # side and take a floored target draw instead.
                                n_floored += 1
                                _fs = client.floored_target_sample(
                                    t_ids, top_k, temperature, floor,
                                    math.exp(max(tmap.values())) if tmap else 0.0, aff + "-t")
                                tid = _fs["sampled_id"]
                                _forced_t_lp = tmap.get(_fs["sampled_str"], float("nan"))
                                _forced_j_lp = jmap.get(_fs["sampled_str"])
                    if _forced_t_lp is not None:
                        t_lp, j_lp = _forced_t_lp, _forced_j_lp     # tid already set above
                    elif fb_mode == "target_sample" and floor > 0.0 and tmap:
                        # target_sample under a floor: redraw with min_p so the target's own
                        # sample cannot land below it either.
                        _fs = client.floored_target_sample(
                            t_ids, top_k, temperature, floor,
                            math.exp(max(tmap.values())), aff + "-t")
                        tid = _fs["sampled_id"]
                        t_lp = tmap.get(_fs["sampled_str"], float("nan"))
                        j_lp = jmap.get(_fs["sampled_str"])
                    elif tid is None:    # target_sample, or the shortcut left tid unset
                        tid, t_lp = tr["sampled_id"], tr["sampled_lp"]
                        j_lp = jmap.get(tr["sampled_str"])
                if tid is None or tid in res.stop_ids:
                    break
                gen.append(tid)
                t_lps.append(t_lp if t_lp is not None else float("nan"))
                j_lps.append(j_lp if j_lp is not None else float("nan"))
                t_ids = t_ids + [tid]
                j_ids = j_ids + [tid]

        # jail_* fallbacks leave holes in t_lps (the emitted token was outside the target's
        # top-k). Re-score the exact id sequence against the target to fill them in; one
        # extra teacher-forced call per turn, and only for these arms.
        # Unconditional, not only when a hole exists: next_topk and echo scoring disagree by
        # ~-0.6pp on the mean, so a transcript mixing both would be measured two ways at once.
        if gen:
            # Mandatory, not best-effort: without it the plausibility mean would be taken
            # over only the tokens the target happened to rank highly, which is precisely
            # the bias this arm is being tested for. _prob_summary also cannot consume a
            # None. Fail loudly rather than emit a flattering or crashing series.
            exact = client.score_ids(t_prefix, gen)
            if len(exact) != len(t_lps):
                raise RuntimeError(
                    f"api_fallback={fb_mode}: rescore returned {len(exact)} logprobs for "
                    f"{len(t_lps)} generated tokens; cannot fill the target probabilities.")
            t_lps = list(exact)

        return {"best_text": _finalize(res, gen), "best_ids": gen,
                "best_token_probs": [math.exp(l) * 100 for l in t_lps],
                "best_token_probs_jail": [(math.exp(l) * 100 if l == l else None) for l in j_lps],
                "n_fallback": n_fallback, "n_unres": n_unres,
                "n_floored": n_floored, "n_resamples": n_resamples,
                "n_shortcut": n_shortcut, "n_greedy": n_greedy, "alpha_sum": alpha_sum,
                "n_empty": n_empty, "q_sum": q_sum,
                "n_mixdrop": n_mixdrop, "n_mixcalls": n_mixcalls,
                "truncated": truncated}

    _t0 = time.time()
    jobs = list(enumerate(target_msgs_batch))
    if len(jobs) == 1:
        out = [_one(jobs[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(len(jobs), 16)) as ex:
            out = list(ex.map(_one, jobs))
    nf = sum(o.pop("n_fallback") for o in out)
    nu = sum(o.pop("n_unres", 0) for o in out)
    nfl = sum(o.pop("n_floored", 0) for o in out)
    nrs = sum(o.pop("n_resamples", 0) for o in out)
    nsc = sum(o.pop("n_shortcut", 0) for o in out)
    ngr = sum(o.pop("n_greedy", 0) for o in out)
    asm = sum(o.pop("alpha_sum", 0.0) for o in out)
    nem = sum(o.pop("n_empty", 0) for o in out)
    qsm = sum(o.pop("q_sum", 0.0) for o in out)
    nmd = sum(o.pop("n_mixdrop", 0) for o in out)
    nmc = sum(o.pop("n_mixcalls", 0) for o in out)
    trunc = [o.pop("truncated") for o in out]
    ncut = sum(1 for x in trunc if x)
    nt = sum(len(o["best_ids"]) for o in out)
    # nf = disjoint top-k sets; nu = overlap non-empty but the pick was unresolvable. Separate
    # counters because they are different events, and nu being ~0 should be checkable.
    print(f"  [api_tilt b1={ob1:g} b2={ob2:g} fb={fb_mode}] {nt} tokens, "
          f"{nf} stage-2 ({100*nf/max(nt,1):.2f}%), {nu} unresolved "
          f"({100*nu/max(nt,1):.2f}%)"
          + (f", {nfl} floored ({100*nfl/max(nt,1):.2f}%)" if floor > 0 else "")
          + (f", {nrs} {'probes' if fb_mode == 'jail_descend' else 'resamples'}" if nrs else "")
          + (f", floor={floor:g} ({nmd} dropped, {nmc} priced)" if floor > 0.0 else "")
          + (f", stage2={stage2_mode} ({nem} empty, mean q={qsm/max(nt,1):.3f})"
             if stage2_mode in ("disagree", "threshold") else "")
          + (f" theta={stage2_theta:g}" if stage2_mode == "threshold" else "")
          + (f", {nsc} argmax-shortcut" if nsc else "")
          + (f", {ngr} greedy ({100*ngr/max(nt,1):.1f}%)" if ngr else "")
          + (f", adaptive alpha0={alpha0:g} k={alpha_k:g} (mean alpha={asm/max(nt,1):.3f})"
             if adaptive else "")
          + (f" q={q_metric}" if q_metric != "elicited_outside" else "")
          + (f", sample T={sample_temp:g}" if sample_temp > 0.0 else "")
          + (f"  |  {ncut}/{len(out)} scenarios CUT SHORT by API failure"
             f" -- e.g. {next(x for x in trunc if x)[:110]}" if ncut else ""), flush=True)
    _record_cost(client, f"mix:{fb_mode}",
                 {"secs": round(time.time() - _t0, 2), "gen_tokens": nt,
                  "n_fallback": nf, "n_unres": nu, "n_floored": nfl, "n_resamples": nrs,
                  "n_shortcut": nsc, "n_empty": nem, "floor": floor,
                  "n_mixdrop": nmd, "n_mixcalls": nmc,
                  "stage2": stage2_mode,
                  "stage2_theta": stage2_theta,
                  "mean_q": round(qsm / max(nt, 1), 4),
                  "n_scenarios": len(out)})
    return out


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
    _rule = str(jail_runtime_cfg.get("api_rule", "corner") or "corner")
    if _rule == "overlap":
        return _driven_overlap(handle, jail_runtime_cfg, target_msgs_batch,
                               max_tokens, temperature, no_think_target)
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
            return {"best_text": _clip(g["text"] or ""), "best_ids": g["ids"],
                    "best_token_probs": probs}
        # Elicited only: sample from the jail context, then score those exact tokens under
        # the TARGET context to get the plausibility metric.
        g = client.generate(_jail_prefix(tm), max_tokens, temperature)
        text, ids = g["text"] or "", g["ids"]
        if not ids:
            return {"best_text": "", "best_ids": [], "best_token_probs": []}
        t_lps = client.score_ids(_target_prefix(tm), ids)
        return {"best_text": _clip(text), "best_ids": ids,
                "best_token_probs": [math.exp(l) * 100 for l in t_lps],
                "best_token_probs_jail": [math.exp(l) * 100 for l in g["logprobs"]]}

    if len(target_msgs_batch) == 1:
        return [_one(target_msgs_batch[0])]
    with ThreadPoolExecutor(max_workers=min(len(target_msgs_batch), 16)) as ex:
        return list(ex.map(_one, target_msgs_batch))


__all__ = ["ApiTiltTarget", "load_api_target", "_jail_generate_api",
           "_driven_overlap"]
