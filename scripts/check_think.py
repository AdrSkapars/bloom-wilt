"""Decide a model's _USES_THINK_BLOCK entry from its chat template, not by guessing.

The registry in core.py records one thing per model: whether the chat template
auto-opens a <think> reasoning block that has to be closed with a prefilled empty
one. Getting it wrong is silent -- the target's whole next-token distribution ends
up dominated by </think> -- so read it off the rendered template instead.

Downloads the tokenizer only, never the weights.

Usage: python scripts/check_think.py <hf-repo> [<hf-repo> ...]
"""

import re
import sys

OPENERS = ("<think>", "<thinking>", "<|thinking|>", "<reasoning>")


def verdict(repo: str) -> None:
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - report whatever the hub/transformers raised
        print(f"{repo}\n  FAILED to load tokenizer: {type(exc).__name__}: {str(exc)[:160]}")
        return

    if not getattr(tok, "chat_template", None):
        print(f"{repo}\n  no chat template -- cannot decide, inspect by hand")
        return

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True
    )
    tail = rendered[-220:]

    opened = [o for o in OPENERS if o in rendered]
    # an auto-opened block is one the template opens and does NOT close before
    # handing over to the model
    dangling = [
        o for o in opened
        if rendered.rfind(o) > rendered.rfind(o.replace("<", "</", 1))
    ]

    print(repo)
    print("  openers present :", opened or "none")
    print("  left unclosed   :", dangling or "none")
    print("  renders as      :", repr(tail))
    key = repo.lower()
    print(f'  registry line   : "{key}": {bool(dangling)},')
    if dangling:
        print("  -> needs the closed-think prefill")
    else:
        print("  -> no wrapper needed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for repo in sys.argv[1:]:
        verdict(repo)
        print()
