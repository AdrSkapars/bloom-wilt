"""Hosted-API rollout stream. Owns `api_jailbroken_output`; Fireworks only.

Forked from rollout.py, which owns `jailbroken_output` and is hf_full only. Neither
accepts the other's engine. Only the batched-jail lockstep branch is carried over -- an
api/ target cannot reach the other three (no logits, no tokenizer, search refused).
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import *
from . import core
from .rollout import _kickoff_message, _build_evaluator_followup, _extract_message_tags
from .apitilt import load_api_target, _jail_generate_api

_FIREWORKS_ONLY = ("fireworks",)


def run_rollout_api(
    cfg: DotDict,
    prompts_yaml: Dict,
    output_dir: Path,
    understanding_results: Dict,
    ideation_results: Dict,
    variations_override: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Rollout for a hosted `api/` target driven by `api_jailbroken_output`."""
    print("\n" + "=" * 60, flush=True)
    print("ROLLOUT STAGE - STARTED (api)", flush=True)
    print("=" * 60, flush=True)

    target_model_id = cfg.rollout.target
    if not target_model_id.startswith("api/"):
        raise RuntimeError(
            f"run_rollout_api requires an 'api/' target, got {target_model_id!r}. "
            f"Local targets belong to jailbroken_output / rollout.py.")
    _provider = (os.environ.get("BLOOM_TARGET_API", "") or "fireworks").strip().lower()
    if _provider not in _FIREWORKS_ONLY:
        raise RuntimeError(
            f"api_jailbroken_output supports Fireworks only (got {_provider!r}): the decode "
            f"needs /completions to accept an integer-array prompt with echo+logprobs.")

    behavior_name = cfg.behavior_name
    behavior_understanding = understanding_results["understanding"]
    scientific_motivation = understanding_results["scientific_motivation"]
    transcript_analyses = understanding_results.get("transcript_analyses", [])
    variations = variations_override if variations_override is not None else ideation_results["variations"]
    max_turns = cfg.rollout.get("max_turns", 3)
    _legacy_mt = cfg.rollout.get("max_tokens", 4000)
    eval_max_tokens = cfg.rollout.get("evaluator_max_tokens", _legacy_mt)
    target_max_tokens = cfg.rollout.get("target_max_tokens", _legacy_mt)
    temperature = cfg.get("temperature", DEFAULT_TEMPERATURE)
    no_think_eval = not cfg.rollout.get("evaluator_thinking", False)
    no_think_target = not cfg.rollout.get("target_thinking", False)
    evaluator_model_id = cfg.rollout.model

    jail_cfg = cfg.get("api_jailbroken_output", {}) or {}
    if not bool(jail_cfg.get("enabled", False)):
        raise RuntimeError(
            "an api/ target requires api_jailbroken_output.enabled=True (the un-steered "
            "corner is target_only=True, not a separate path).")
    for _dead in ("search_input", "search_output", "flrt_search_input", "tokbias_output"):
        if bool((cfg.get(_dead, {}) or {}).get("enabled", False)):
            raise RuntimeError(
                f"{_dead} is enabled but the target is hosted: an api/ target exposes no "
                f"logits or tokenizer, so token-level search is impossible. Disable it.")
    if bool((cfg.get("jailbroken_output", {}) or {}).get("enabled", False)):
        raise RuntimeError(
            "jailbroken_output is enabled alongside an api/ target. That section is the "
            "paper's hf_full LogitTilt; use api_jailbroken_output instead.")

    _target_only = bool(jail_cfg.get("target_only", False))
    jail_system_prompt = prompts_yaml.get("jailbroken_output_system_prompt", "")
    if not jail_system_prompt and not _target_only:
        raise RuntimeError(
            "api_jailbroken_output.enabled=True requires 'jailbroken_output_system_prompt' "
            "in the behaviour yaml.")
    jail_runtime_cfg = {
        "engine": "api_tilt",
        "enabled": True,
        "target_only": _target_only,
        "system_prompt": jail_system_prompt,
        "prefill": (prompts_yaml.get("jailbroken_output_prefill", "") or "") if jail_cfg.get("prefill", True) else "",
        "b1": (float(jail_cfg["b1"]) if jail_cfg.get("b1") is not None else 1.0),
        "b2": float(jail_cfg.get("b2", 1.0)),
        "target_floor": 0.0,   # needs full-vocab target logits; impossible over a text API
        "api_rule": str(jail_cfg.get("rule", "corner") or "corner"),
        "api_pick": str(jail_cfg.get("pick", "elicited") or "elicited"),
        "api_fallback": str(jail_cfg.get("fallback", "target_sample") or "target_sample"),
        "api_beta": float(jail_cfg.get("beta", 1.0) or 1.0),
        "api_top_k": int(jail_cfg.get("top_k", 5) or 5),
        "api_fb_floor": float(jail_cfg.get("fb_floor", 0.0) or 0.0),
        "api_fb_tries": int(jail_cfg.get("fb_tries", 5) or 5),
    }

    if evaluator_model_id.startswith("local/"):
        lm_eval = _get_local_model(
            evaluator_model_id[len("local/"):],
            gpu_id=int(os.environ.get("BLOOM_EVAL_GPU", cfg.get("evaluator_gpu_id", 0))),
            gpu_memory_utilization=float(os.environ.get(
                "BLOOM_EVAL_UTIL", cfg.get("evaluator_gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTIL))),
            max_model_len=core._DEFAULT_MAX_MODEL_LEN)
    else:
        lm_eval = ApiModel(evaluator_model_id, max_tokens=eval_max_tokens,
                           reasoning_effort=_effort(cfg.rollout.get("evaluator_thinking", False)))
        print(f"  [evaluator] hosted API model {evaluator_model_id!r} via litellm "
              f"(reasoning_effort={lm_eval.reasoning_effort!r}) -- sampling evaluator turns only",
              flush=True)

    # one hosted handle serves both contexts (self-jail; no weights)
    jail_runtime_cfg["hf"] = load_api_target(target_model_id)
    print(f"  [api_jailbroken_output] target={target_model_id} "
          f"rule={jail_runtime_cfg['api_rule']} pick={jail_runtime_cfg['api_pick']} "
          f"fb={jail_runtime_cfg['api_fallback']} beta={jail_runtime_cfg['api_beta']:g} "
          f"(b1={jail_runtime_cfg['b1']}, b2={jail_runtime_cfg['b2']})", flush=True)

    evaluator_system_prompt = build_rollout_system(behavior_name, prompts_yaml)
    target_sysprompt_prefix = _get_override(prompts_yaml, "target_sysprompt_prefix")
    target_kickoff_prefix = _get_override(prompts_yaml, "target_kickoff_prefix")
    generate_kickoff_additional = _get_override(prompts_yaml, "generate_kickoff_additional")

    def _build_kickoff_prompt(refine_context: str = "") -> str:
        return _kickoff_message(generate_kickoff_additional, refine_context)

    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    rollouts: List[Dict] = []
    beast_pool_data: List[Dict] = []

    # the kickoff prompt. setup_content stays as an unused field (always "") for
    # back-compat with downstream code that reads it from transcript metadata.
    var_descs: List[str] = []
    rollout_prompt_texts: List[str] = []

    # Resume: figure out which variations are already complete on disk so we can skip them
    # entirely (no input search, no rollout). A variation counts as complete if its single
    # transcript rep (transcript_v{var_idx}r1.json) exists.
    def _variation_done(var_idx_1based: int) -> bool:
        return (transcripts_dir / f"transcript_v{var_idx_1based}r1.json").exists()

    n_skipped = 0
    for var_idx_0based, variation in enumerate(variations):
        var_idx = var_idx_0based + 1
        vd = variation.get("description", str(variation)) if isinstance(variation, dict) else str(variation)
        var_descs.append(vd)
        if _variation_done(var_idx):
            rollout_prompt_texts.append("")
            n_skipped += 1
            continue
        # Drop scientific_motivation in round 2+ when refine_context is doing the heavy
        # lifting — it duplicates the high-level framing already covered by
        # behavior_understanding and the history injected via the kickoff.
        has_refine_context = bool(
            variation.get("refine_context", "") if isinstance(variation, dict) else ""
        )
        rp = build_rollout_prompt(
            behavior_name, behavior_understanding, scientific_motivation,
            transcript_analyses, vd, max_turns, prompts_yaml,
            skip_motivation=has_refine_context,
        )
        rollout_prompt_texts.append(rp)

    if n_skipped:
        print(f"  Resume: {n_skipped}/{len(variations)} variations already have transcripts — skipping", flush=True)
    print(f"  Setup-generation pass disabled — using fixed target_system_prompt from the yaml", flush=True)
    setup_contents: List[str] = [""] * len(variations)

    def _resume_load_variation(var_idx: int, var_desc: str) -> None:
        """Load an already-complete variation's transcript into `rollouts` (resume path)."""
        for rep in (1,):
            tf_path = transcripts_dir / f"transcript_v{var_idx}r{rep}.json"
            try:
                with open(tf_path, "r", encoding="utf-8") as f:
                    td = json.load(f)
            except Exception as e:
                print(f"    Could not read existing {tf_path.name}: {e} — will be missing from rollout summary", flush=True)
                continue
            turn_lps = [m["targeted_response_start_logprob"] for m in td.get("messages", [])
                        if m.get("targeted_response_start_logprob") is not None]
            avg_lp = round(sum(turn_lps) / len(turn_lps), 4) if turn_lps else None
            entry: Dict[str, Any] = {
                "variation_number": var_idx, "variation_description": var_desc,
                "repetition_number": rep, "num_turns": len(td.get("messages", [])),
                "transcript_file": tf_path.name,
            }
            if avg_lp is not None:
                entry["avg_logprob"] = avg_lp
            rollouts.append(entry)

    # ══ Batched JAIL rollout (variations in LOCKSTEP) ════════════════════
    # Mirror of the corruption batched path for the jail hf_full PoE decode. The
    # gate guarantees no input/output token-search, so each turn is just: sample an
    # evaluator message (vLLM, batched natively) then generate the target reply via
    # jail PoE. _jail_generate_hf batches the whole active chunk in ONE call per turn
    # (B slots), far more GPU-efficient than the per-variation serial path below.
    jail_var_batch = max(1, int(os.environ.get("BLOOM_API_JAIL_VAR_BATCH",
                                (cfg.get("api_jailbroken_output", {}) or {}).get("var_batch", 12))))
    _jail_hf = jail_runtime_cfg["hf"]

    # One "seed" per transcript (variation x rep), honoring resume/skip. freeze_input
    # rounds 2+ replicate a variation into n_reps seeds that share the frozen kickoff
    # but resample the target output independently.
    seeds = []
    for var_idx, (variation, var_desc, rollout_prompt_text, _sc) in enumerate(
        zip(variations, var_descs, rollout_prompt_texts, setup_contents), 1
    ):
        if _variation_done(var_idx):
            print(f"  Variation {var_idx}/{len(variations)}: skipped (transcripts exist)", flush=True)
            _resume_load_variation(var_idx, var_desc)
            continue
        fixed_kickoff = variation.get("fixed_kickoff") if isinstance(variation, dict) else None
        frozen = bool(fixed_kickoff and fixed_kickoff.get("content"))
        n_reps = max(1, int(variation.get("n_reps", 1))) if (frozen and isinstance(variation, dict)) else 1
        for rep in range(1, n_reps + 1):
            seeds.append({"var_idx": var_idx, "variation": variation, "var_desc": var_desc,
                          "rollout_prompt_text": rollout_prompt_text, "rep": rep,
                          "frozen": frozen, "fixed_kickoff": fixed_kickoff})

    # ── Kickoff for ALL seeds up front in ONE batched evaluator call ──
    gen_seeds, gen_ctxs = [], []
    for sd in seeds:
        variation = sd["variation"]
        frozen_tsp = variation.get("target_system_prompt", "") if isinstance(variation, dict) else ""
        target_sysprompt = frozen_tsp or prompts_yaml.get("target_system_prompt", "")
        if target_sysprompt_prefix and target_sysprompt_prefix.strip() and target_sysprompt and not frozen_tsp:
            target_sysprompt = f"{target_sysprompt_prefix.strip()}\n\n{target_sysprompt}"
        per_var_refine_context = variation.get("refine_context", "") if isinstance(variation, dict) else ""
        kickoff_prompt = _build_kickoff_prompt(refine_context=per_var_refine_context)
        eval_ctx = [
            {"role": "system", "content": evaluator_system_prompt},
            {"role": "user", "content": f"{sd['rollout_prompt_text']}\n\n{kickoff_prompt}"},
        ]
        sd["target_sysprompt"] = target_sysprompt
        sd["eval_kickoff_ctx"] = eval_ctx
        sd["kickoff_strategy"] = ""
        if sd["frozen"]:
            sd["kickoff_msg"] = sd["fixed_kickoff"]["content"]
            sd["kickoff_strategy"] = sd["fixed_kickoff"].get("strategy", "") or ""
        else:
            gen_seeds.append(sd); gen_ctxs.append(_strip_thinking_from_msgs(eval_ctx))
    if gen_ctxs:
        raws = batch_generate_local(lm_eval, gen_ctxs, eval_max_tokens, temperature, no_think=no_think_eval)
        for sd, raw in zip(gen_seeds, raws):
            parsed = parse_message(_make_local_response(raw))
            content = parsed["content"] or raw
            msg, _trs, strat = _extract_message_tags(content)
            sd["kickoff_msg"] = msg
            sd["kickoff_strategy"] = strat

    # ── Roll out mini-batches (chunks) in lockstep — ONE jail PoE call per turn ──
    for _b in range(0, len(seeds), jail_var_batch):
        chunk = seeds[_b:_b + jail_var_batch]
        for sd in chunk:
            tsp = sd["target_sysprompt"]
            tmsgs, trmsgs = [], []
            if tsp:
                tmsgs.append({"role": "system", "content": tsp})
                trmsgs.append({"role": "system", "content": tsp, "source": "target_system"})
            kmsg = sd["kickoff_msg"]
            target_content = kmsg
            if target_kickoff_prefix and not sd["frozen"]:
                target_content = target_kickoff_prefix.strip() + " " + kmsg
            tmsgs.append({"role": "user", "content": target_content})
            kick_entry = {"role": "user", "content": target_content, "source": "evaluator"}
            if sd["kickoff_strategy"]:
                kick_entry["strategy"] = sd["kickoff_strategy"]
            trmsgs.append(kick_entry)
            sd["target_msgs"] = tmsgs
            sd["transcript_msgs"] = trmsgs
            sd["eval_msgs"] = list(sd["eval_kickoff_ctx"]) + [{"role": "assistant", "content": kmsg}]
            sd["current_turn"] = 0
            sd["done"] = False
            beast_pool_data.append({
                "variation_number": sd["var_idx"], "turn": "kickoff", "trs": "",
                "pool": [{"baseline": kmsg, "suffix": "", "message": kmsg, "score": None}],
            })

        for turn in range(max_turns):
            active = [sd for sd in chunk if not sd["done"]]
            if not active:
                break
            jail_results = _jail_generate_api(
                _jail_hf, jail_runtime_cfg,
                [sd["target_msgs"] for sd in active], target_max_tokens, temperature, no_think_target,
            )
            for sd, _jr in zip(active, jail_results):
                raw_target = _jr["best_text"]
                _ids = _jr.get("best_ids")
                _tprobs = _jr.get("best_token_probs")
                parsed_t = parse_message(_make_local_response(raw_target))
                target_resp = parsed_t["content"] or raw_target
                target_reason = parsed_t["reasoning"]
                sd["target_msgs"].append({"role": "assistant", "content": target_resp})
                sd["current_turn"] = turn + 1
                tmsg = {"role": "assistant", "content": target_resp, "source": "target"}
                if _ids and target_resp == raw_target:
                    tmsg["gen_token_ids"] = _ids           # exact ids -> free token stats
                if _tprobs and target_resp == raw_target:
                    tmsg["gen_token_probs"] = _tprobs      # on-policy probs (plausibility)
                if _tprobs:
                    tmsg["prob_stats"] = _prob_summary(_tprobs)  # summary valid even when parse strips channel/reasoning markers (e.g. gemma-4)
                _jprobs = _jr.get("best_token_probs_jail")       # api_tilt elicited-only: probs under the JAIL context the tokens were drawn from
                if _jprobs and target_resp == raw_target:
                    tmsg["gen_token_probs_jail"] = _jprobs
                if target_reason:
                    tmsg["reasoning"] = target_reason
                sd["transcript_msgs"].append(tmsg)
                if sd["current_turn"] >= max_turns:
                    sd["done"] = True

            cont = [sd for sd in chunk if not sd["done"]]
            if not cont:
                break
            gen_ctxs = []
            for sd in cont:
                last = sd["transcript_msgs"][-1]
                followup = _build_evaluator_followup(
                    last["content"], last.get("reasoning"), sd["current_turn"], max_turns,
                )
                eval_msgs_turn = list(sd["eval_msgs"]) + [{"role": "user", "content": followup}]
                sd["_eval_msgs_turn"] = eval_msgs_turn
                gen_ctxs.append(_strip_thinking_from_msgs(eval_msgs_turn))
            raws = batch_generate_local(lm_eval, gen_ctxs, eval_max_tokens, temperature, no_think=no_think_eval)
            for sd, raw in zip(cont, raws):
                parsed = parse_message(_make_local_response(raw))
                content = parsed["content"] or raw
                next_msg, _trs, strat = _extract_message_tags(content)
                if "<END>" in next_msg:
                    sd["done"] = True
                    continue
                sd["eval_msgs"] = sd["_eval_msgs_turn"] + [{"role": "assistant", "content": next_msg}]
                sd["target_msgs"].append({"role": "user", "content": next_msg})
                turn_entry = {"role": "user", "content": next_msg, "source": "evaluator"}
                if strat:
                    turn_entry["strategy"] = strat
                sd["transcript_msgs"].append(turn_entry)

        # ── Save transcripts for this chunk ──
        for sd in chunk:
            var_idx, rep = sd["var_idx"], sd["rep"]
            transcript_data = {
                "metadata": {
                    "evaluator_model": evaluator_model_id,
                    "target_model": target_model_id,
                    "target_system_prompt": sd["target_sysprompt"],
                    "setup_content": "",
                    "variation_number": var_idx,
                    "repetition_number": rep,
                    "created_at": datetime.now().isoformat(),
                },
                "messages": sd["transcript_msgs"],
                "judgment": None,
            }
            filename = f"transcript_v{var_idx}r{rep}.json"
            save_json(transcript_data, transcripts_dir / filename)
            print(f"  Rollout v{var_idx}r{rep} done ({sd['current_turn']} turns) [batched-jail]", flush=True)
            rollouts.append({
                "variation_number": var_idx, "variation_description": sd["var_desc"],
                "repetition_number": rep, "num_turns": len(sd["transcript_msgs"]),
                "transcript_file": filename, "kickoff_score": None,
            })


    save_json({"beast_pools": beast_pool_data}, output_dir / "beast_pool.json")
    rollouts.sort(key=lambda x: (x["variation_number"], x["repetition_number"]))
    all_lps = [r["avg_logprob"] for r in rollouts if r.get("avg_logprob") is not None]
    mean_avg_logprob = round(sum(all_lps) / len(all_lps), 4) if all_lps else None

    rollout_results = {
        "metadata": {
            "evaluator": evaluator_model_id,
            "target":    target_model_id,
            "max_turns": max_turns,
            "api_jailbroken_output": {k: v for k, v in jail_runtime_cfg.items()
                                      if k != "hf"},   # handle is not serialisable
        },
        "rollouts":        rollouts,
        "successful_count": len(rollouts),
        "failed_count":    0,
        "total_count":     len(variations),
        "logprob_summary": {
            "mean_avg_logprob": mean_avg_logprob,
            "num_scored":       len(all_lps),
        },
    }
    # Token-prob stats of the chosen outputs, computed here while the corruption target model
    # is still loaded (reused — no reload). Stashed for the judgment summary + rollout.json.
    #   • target_only (baseline/BoN) → probs captured DURING the HF decode, read from the
    #     transcript — no extra forward pass.
    #   • corruption ON → score the exact PoE outputs with the loaded HF corruption target.
    # Both report the same token metrics (same HF target model, same summary).
    _tok = None
    try:
        _tok = token_stats_from_stored(output_dir)   # on-policy probs captured during the (jail/BoN) HF decode, read from the transcript
    except Exception as e:
        print(f"  [token stats] skipped: {e}", flush=True)
    if _tok:
        rollout_results["token_stats"] = _tok
        print(f"  Token probs: avg={_tok['A_mean_tok_pct']:.2f}% | mean-of-mins="
              f"{_tok['B_mean_of_mins_pct']:.3f}% | min-of-mins={_tok['B_min_of_mins_pct']:.5f}%", flush=True)
    save_json(rollout_results, output_dir / "rollout.json")
    if mean_avg_logprob is not None:
        print(f"  Mean avg logprob: {mean_avg_logprob:.4f} over {len(all_lps)} transcripts", flush=True)
    print(f"ROLLOUT STAGE - COMPLETED ({len(rollouts)} rollouts)", flush=True)
    return rollout_results


__all__ = ['run_rollout_api']
