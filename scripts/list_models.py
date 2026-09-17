"""Print the model ids your Gemini API key can actually use.

Model names change between releases; run this once and copy the id you want
into GEMINI_MODEL in .env.

    python scripts/list_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def main() -> int:
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set in .env — nothing to list.", file=sys.stderr)
        return 1

    try:
        from google import genai
    except ImportError:
        print("google-genai is not installed. Run: pip install google-genai", file=sys.stderr)
        return 1

    client = genai.Client(api_key=settings.gemini_api_key)
    print(f"Models available to this key (current GEMINI_MODEL={settings.gemini_model}):\n")

    found = False
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        found = True
        marker = "  <-- currently configured" if settings.gemini_model in model.name else ""
        print(f"  {model.name}{marker}")

    if not found:
        print("  (none returned — check the key's permissions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
