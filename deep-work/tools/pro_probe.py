#!/usr/bin/env python3
"""Verify one authorized Pro API route. Default is dry run; not an agent harness."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Any


def make_request(model: str, effort: str) -> dict[str, Any]:
    return {"model": model, "reasoning": {"mode": "pro", "effort": effort},
            "input": "Configuration probe only. Reply exactly PRO_PROBE_OK.",
            "max_output_tokens": 2048}  # Probe-only cap; not the research budget.


def make_receipt(response: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    reasoning = response.get("reasoning") or {}
    mode = reasoning.get("mode") if isinstance(reasoning, dict) else None
    checked = datetime.now(timezone.utc).isoformat()
    return {"checked_utc": checked, "checked_at_utc": checked,
            "response_id": response.get("id"), "model": response.get("model"),
            "model_requested": request["model"], "model_returned": response.get("model"),
            "requested_mode": request["reasoning"]["mode"],
            "effective_mode": mode, "requested_effort": request["reasoning"]["effort"],
            "returned_effort": reasoning.get("effort") if isinstance(reasoning, dict) else None,
            "evidence_kind": "provider_metadata" if mode is not None else "requested_only",
            "response_status": response.get("status"), "usage": response.get("usage"),
            "pro_verified": mode == "pro",
            "scope": "This call only; no claim about other workers or future calls."}


# v2 name; keep make_receipt and legacy receipt fields for existing callers.
receipt_from = make_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize one potentially paid probe")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="max", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--out", type=Path, default=Path("pro-receipt.json"))
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    req_data = make_request(args.model, args.effort)
    if not args.live:
        print("DRY RUN: no network request or charge.")
        print(json.dumps(req_data, indent=2))
        return 0
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.out.exists():
        print("Refusing to overwrite an existing receipt.", file=sys.stderr)
        return 2
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("No OPENAI_API_KEY found. No request made. Use an already authorized credential route.", file=sys.stderr)
        return 2
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses", data=json.dumps(req_data).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as stream:
            response = json.load(stream)
        if not isinstance(response, dict):
            raise ValueError("Unexpected response format")
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}; no fallback or automatic retry.", file=sys.stderr)
        return 3
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"Probe failed ({type(exc).__name__}). A timeout does not establish server completion or billing; inspect before retrying.", file=sys.stderr)
        return 3
    receipt = make_receipt(response, req_data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except FileExistsError:
        print("Receipt destination appeared during call; refusing overwrite. Response ID: " + str(response.get("id")), file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    if not receipt["pro_verified"]:
        print("Pro was requested but effective mode is not verified. No silent fallback.", file=sys.stderr)
        return 4
    return 0 if response.get("status") == "completed" else 5


if __name__ == "__main__":
    raise SystemExit(main())
