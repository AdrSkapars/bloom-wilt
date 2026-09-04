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
        for i in range(self._tok.get_vocab_size()):
            s = self._tok.decode([i], skip_special_tokens=False)
            if s not in self._by_text:      # 36 byte-fallback strings collide; keep the lowest id
                self._by_text[s] = i
        self.eos_id = self._tok.get_vocab().get("<｜end▁of▁sentence｜>", 1)

    def id_of(self, text: str) -> Optional[int]:
        return self._by_text.get(text)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)


def _wsample(items, key):
    """Sample one item with weight proportional to exp(key(item)).

    Shifted by the max before exponentiating: `combined` scores are sums of two logprobs,
    so exp() of them underflows to zero well inside the range we actually see.
    """
    ls = [key(x) for x in items]
    m = max(ls)
    w = [math.exp(l - m) for l in ls]
    tot = sum(w) or 1.0
    r, acc = random.random() * tot, 0.0
    for x, wi in zip(items, w):
        acc += wi
        if r <= acc:
            return x
    return items[-1]


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
        from jinja2 import Environment
        self._tpl = Environment().from_string(Path(template_path).read_text(encoding="utf-8"))
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
                  affinity: Optional[str] = None) -> Dict:
        """One decode step: the top-`top_k` candidates for the position after `ids`.

        Asks for a single token so the response carries `top_logprobs[0]` — the
        distribution at that position — plus the token the provider itself sampled from
        the full (untruncated) distribution, which the empty-overlap fallback uses
        directly. `top_p`/`top_k` are pinned wide open so that sample is a genuine draw
        from the whole distribution, not a truncated one.

        Returns {"top": [(token_string, logprob) x top_k], "sampled_id", "sampled_str",
        "sampled_lp", "cached"}.
        """
        r = self._post({"model": self.model, "prompt": list(ids), "max_tokens": 1,
                        "temperature": float(temperature), "top_p": 1.0, "top_k": 0,
                        "logprobs": int(top_k), "echo": False}, affinity=affinity)
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
    # "argmax" is the rule as specified: the overlap member with the highest elicited
    # probability. It is deterministic, so every round reproduces the same transcript --
    # "sample" draws from the overlap in proportion to the elicited probabilities instead,
    # which keeps round-to-round diversity for pools and post-run selection.
    pick_mode = str(jail_runtime_cfg.get("api_pick", "elicited") or "elicited")
    # What to emit when the two top-k sets are DISJOINT. "target_sample" is a plain draw
    # from the target's full distribution (the provider already sampled one, so it costs
    # nothing); the top5_* variants restrict that choice to the target's own top-k instead.
    # Only fires on ~1-6% of positions, so it is a small lever by construction.
    fb_mode = str(jail_runtime_cfg.get("api_fallback", "target_sample") or "target_sample")
    # Weight on the ELICITED term of the combined score: z = l_target + beta * l_elicited,
    # restricted to the overlap. This is the tilt's own b2 knob, reintroduced on the top-k
    # candidate set; beta=1 is the plain product of the two probabilities. Under
    # combined_sample it also sharpens the draw, exactly as b2 does on the local path.
    beta = float(jail_runtime_cfg.get("api_beta", 1.0) or 1.0)

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

        gen, t_lps, j_lps, n_fallback = [], [], [], 0
        with ThreadPoolExecutor(max_workers=2) as ex:
            for _ in range(int(max_tokens)):
                ft = ex.submit(client.next_topk, t_ids, top_k, temperature, aff + "-t")
                fj = ex.submit(client.next_topk, j_ids, top_k, temperature, aff + "-j")
                tr, jr = ft.result(), fj.result()
                tmap = dict(tr["top"])
                overlap = [(s, lp) for s, lp in jr["top"] if s in tmap]
                # z = l_target + beta * l_elicited, over this step's candidates only
                _comb = lambda x, _m=tmap: _m[x[0]] + beta * x[1]
                if overlap:
                    # overlap entries are (token_string, ELICITED logprob); tmap holds the
                    # TARGET logprob for the same strings.
                    if pick_mode in ("elicited", "argmax"):
                        pick = max(overlap, key=lambda x: x[1])[0]
                    elif pick_mode == "target":
                        pick = max(overlap, key=lambda x: tmap[x[0]])[0]
                    elif pick_mode == "combined":
                        # Summing logprobs multiplies the probabilities — the top-5-restricted
                        # form of the true tilt at b1=1, b2=beta, argmaxed rather than sampled.
                        pick = max(overlap, key=_comb)[0]
                    elif pick_mode == "combined_min":
                        # Anti-selection control: the LEAST probable overlap member.
                        pick = min(overlap, key=_comb)[0]
                    elif pick_mode == "combined_sample":
                        # Draw from the overlap in proportion to the PRODUCT of the two
                        # probabilities, rather than argmaxing it. Unlike "random" this
                        # respects the ranking, and unlike "combined" it is stochastic, so
                        # rounds differ and a pool exists for post-run selection.
                        pick = _wsample(overlap, _comb)[0]
                    elif pick_mode == "random":
                        pick = random.choice(overlap)[0]
                    elif pick_mode == "sample":
                        w = [math.exp(lp) for _, lp in overlap]
                        tot = sum(w) or 1.0
                        r, acc = random.random() * tot, 0.0
                        pick = overlap[-1][0]
                        for (s, _), wi in zip(overlap, w):
                            acc += wi
                            if r <= acc:
                                pick = s
                                break
                    else:
                        raise RuntimeError(
                            f"jailbroken_output.api_pick={pick_mode!r} unknown "
                            f"(elicited | target | combined | random | sample)")
                    tid = res.id_of(pick)
                    if tid is None:                    # unresolvable surface form
                        tid, t_lp = tr["sampled_id"], tr["sampled_lp"]
                        j_lp = dict(jr["top"]).get(tr["sampled_str"])
                        n_fallback += 1
                    else:
                        t_lp = tmap[pick]
                        j_lp = dict(jr["top"])[pick]
                else:
                    n_fallback += 1
                    jmap = dict(jr["top"])
                    cand = None
                    if fb_mode == "top5_argmax" and tr["top"]:
                        cand = tr["top"][0][0]
                    elif fb_mode == "top5_random" and tr["top"]:
                        cand = random.choice(tr["top"])[0]
                    elif fb_mode == "top5_weighted" and tr["top"]:
                        cand = _wsample(tr["top"], lambda x: x[1])[0]
                    elif fb_mode not in ("target_sample", "top5_argmax", "top5_random",
                                         "top5_weighted"):
                        raise RuntimeError(
                            f"jailbroken_output.api_fallback={fb_mode!r} unknown (target_sample "
                            f"| top5_argmax | top5_random | top5_weighted)")
                    cid = res.id_of(cand) if cand is not None else None
                    if cid is None:      # target_sample, or an unresolvable surface form
                        tid, t_lp = tr["sampled_id"], tr["sampled_lp"]
                        j_lp = jmap.get(tr["sampled_str"])
                    else:
                        tid, t_lp, j_lp = cid, tmap[cand], jmap.get(cand)
                if tid is None or tid == res.eos_id:
                    break
                gen.append(tid)
                t_lps.append(t_lp)
                j_lps.append(j_lp if j_lp is not None else float("nan"))
                t_ids = t_ids + [tid]
                j_ids = j_ids + [tid]

        return {"best_text": res.decode(gen).strip(), "best_ids": gen,
                "best_token_probs": [math.exp(l) * 100 for l in t_lps],
                "best_token_probs_jail": [(math.exp(l) * 100 if l == l else None) for l in j_lps],
                "n_fallback": n_fallback}

    jobs = list(enumerate(target_msgs_batch))
    if len(jobs) == 1:
        out = [_one(jobs[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(len(jobs), 16)) as ex:
            out = list(ex.map(_one, jobs))
    nf = sum(o.pop("n_fallback") for o in out)
    nt = sum(len(o["best_ids"]) for o in out)
    print(f"  [api_tilt rule=overlap pick={pick_mode} beta={beta:g} fb={fb_mode}] {nt} tokens, "
          f"{nf} from the empty-overlap fallback ({100*nf/max(nt,1):.1f}%)", flush=True)
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
    if str(jail_runtime_cfg.get("api_rule", "corner") or "corner") == "overlap":
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


__all__ = ["ApiTiltTarget", "load_api_target", "_jail_generate_api", "_driven_overlap"]
