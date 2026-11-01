#!/usr/bin/env python3
"""Optional, explicit paid-call probe. Not a research agent or orchestration harness.

No request is made unless --live is supplied. Credentials are read from the
process environment and are never written to the receipt. Run this process
separately from untrusted experimental code.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_request(model: str, effort: str) -> dict[str, Any]:
    return {
        "model": model,
        "reasoning": {"mode": "pro", "effort": effort},
        "input": "This is a configuration connectivity probe. Reply exactly PRO_PROBE_OK.",
        "max_output_tokens": 2048,
    }


def receipt_from(response: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    reasoning = response.get("reasoning") or {}
    mode = reasoning.get("mode")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_id": response.get("id"),
        "model_requested": request["model"],
        "model_returned": response.get("model"),
        "requested_mode": request["reasoning"]["mode"],
        "effective_mode": mode,
        "requested_effort": request["reasoning"]["effort"],
        "returned_effort": reasoning.get("effort"),
        "response_status": response.get("status"),
        "usage": response.get("usage"),
        "evidence_kind": "provider_metadata" if mode is not None else "requested_only",
        "pro_verified": mode == "pro",
        "scope": "This request only; does not verify other workers, tools, or future requests.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly authorize one paid API probe")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="max", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--out", type=Path, default=Path("pro-receipt.json"))
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout in seconds; not a task-duration estimate")
    args = parser.parse_args()
    request = make_request(args.model, args.effort)
    if not args.live:
        print("DRY RUN: no network request and no charge.")
        print(json.dumps(request, indent=2))
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.out.exists():
        print("Refusing to overwrite an existing receipt. Choose a new --out path.", file=sys.stderr)
        return 2
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY is not set. No request made. Use your authorized local credential mechanism.", file=sys.stderr)
        return 2
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"API returned HTTP {exc.code}. No fallback or automatic retry was attempted.", file=sys.stderr)
        return 3
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"Probe failed ({type(exc).__name__}). A timeout does not establish whether the server billed/completed it; inspect the account before retrying.", file=sys.stderr)
        return 3
    receipt = receipt_from(result, request)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except FileExistsError:
        print("Receipt destination appeared during request. Refusing overwrite; response ID: " + str(result.get("id")), file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    if not receipt["pro_verified"]:
        print("Mode not verified. Requested Pro alone is insufficient; no silent High fallback.", file=sys.stderr)
        return 4
    if result.get("status") != "completed":
        print("Pro metadata was returned but the probe did not complete normally.", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
