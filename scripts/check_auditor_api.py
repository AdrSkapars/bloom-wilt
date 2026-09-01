"""Check the auditor routes through OpenRouter before paying for a box.

Prompts for the key without echoing it, saves it to .env.local (gitignored), then
makes one small call over the same litellm path BLOOM_EVAL_MODEL uses, so a
routing failure shows up here rather than at $8/hour.

Usage: python scripts/check_auditor_api.py
"""

import getpass
import os
import sys
from pathlib import Path

MODEL = os.environ.get("BLOOM_EVAL_MODEL") or "openrouter/google/gemma-4-26b-a4b-it"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env.local"


def _key() -> str:
    for line in (ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("OPENROUTER_API_KEY") or getpass.getpass(
        "OpenRouter API key (nothing will appear): "
    ).strip()


def main() -> int:
    key = _key()
    if not key:
        print("no key given")
        return 1
    os.environ["OPENROUTER_API_KEY"] = key

    if "OPENROUTER_API_KEY" not in (
        ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    ):
        with ENV_FILE.open("a", encoding="utf-8") as fh:
            fh.write("OPENROUTER_API_KEY=" + key + "\n")
        print("saved to", ENV_FILE.name, "(gitignored)")

    import litellm

    litellm.suppress_debug_info = True
    print("model :", MODEL)
    try:
        reply = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: ROUTING OK"}],
            max_tokens=16,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - report whatever litellm raised
        print("FAIL  :", type(exc).__name__)
        print("       ", str(exc)[:400])
        return 1

    print("reply :", (reply.choices[0].message.content or "").strip()[:100])
    print("usage :", getattr(reply, "usage", None))
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
