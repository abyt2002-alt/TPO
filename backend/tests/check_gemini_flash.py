#!/usr/bin/env python3
"""
Quick connectivity test for Gemini 2.5 Flash.

Why this exists:
- Confirms whether the backend machine can reach Gemini endpoint.
- Confirms API key works.
- Prints a short response preview on success.

Usage:
  python backend/tests/check_gemini_flash.py

Optional env vars:
  GEMINI_API_KEY / GOOGLE_API_KEY  (required)
  GEMINI_MODEL                      (default: gemini-2.5-flash)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    model = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    if not api_key:
        print("FAIL: GEMINI_API_KEY/GOOGLE_API_KEY is not set.")
        return 2

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Reply with exactly one short line: "
                            "'Gemini connectivity test OK'."
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.9,
            "maxOutputTokens": 64,
        },
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            body = resp.read().decode("utf-8")
        elapsed = (time.perf_counter() - start) * 1000.0
        parsed = json.loads(body)

        preview = ""
        for candidate in parsed.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                text = str(part.get("text") or "").strip()
                if text:
                    preview = text
                    break
            if preview:
                break

        if not preview:
            print("FAIL: Request succeeded but response text is empty.")
            print("Raw keys:", ", ".join(sorted(parsed.keys())))
            return 3

        print(f"PASS: Gemini reachable. Model={model}. Latency={elapsed:.0f} ms")
        print(f"Response: {preview[:200]}")
        return 0

    except urllib.error.HTTPError as err:
        try:
            detail = err.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = str(err)
        print(f"FAIL: HTTP {err.code} from Gemini.")
        print(f"Detail: {detail[:400]}")
        return 4
    except Exception as err:
        print(f"FAIL: Network/runtime error: {err}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())

