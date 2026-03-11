"""Strict snapshot parity harness for backend API responses.

Usage:
  python backend/tests/parity/run_snapshot_parity.py --mode capture
  python backend/tests/parity/run_snapshot_parity.py --mode compare

Capture mode:
  Calls all requests from requests.json and writes snapshots.json.

Compare mode:
  Calls all requests from requests.json and compares against snapshots.json.
  Rules:
  - Exact key equality for dicts
  - Exact list lengths and ordering
  - Float tolerance of 1e-9
  - Datetime strings compared exactly as strings
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REQUESTS = Path(__file__).resolve().parent / "requests.json"
DEFAULT_SNAPSHOTS = Path(__file__).resolve().parent / "snapshots.json"
FLOAT_TOL = 1e-9


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def do_http_request(base_url: str, case: Dict[str, Any]) -> Dict[str, Any]:
    method = str(case.get("method", "POST")).upper()
    path = str(case["path"])
    payload = case.get("payload", {})
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
            body = json.loads(raw) if raw else None
            return {"status": int(resp.status), "body": body}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8") if e.fp else ""
        try:
            body = json.loads(raw) if raw else None
        except Exception:
            body = raw
        return {"status": int(e.code), "body": body}
    except urllib.error.URLError as e:
        raise RuntimeError(f"request failed for {url}: {e}") from e


def capture(base_url: str, requests_file: Path) -> Dict[str, Any]:
    spec = load_json(requests_file)
    cases = spec.get("cases", [])
    out: Dict[str, Any] = {"cases": {}}
    for case in cases:
        name = case["name"]
        out["cases"][name] = do_http_request(base_url, case)
        print(f"[capture] {name}: status={out['cases'][name]['status']}")
    return out


def compare_values(a: Any, b: Any, path: str, errors: List[str]) -> None:
    if type(a) != type(b):
        errors.append(f"{path}: type mismatch {type(a).__name__} != {type(b).__name__}")
        return

    if isinstance(a, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if a_keys != b_keys:
            missing_in_actual = sorted(list(b_keys - a_keys))
            missing_in_expected = sorted(list(a_keys - b_keys))
            errors.append(
                f"{path}: key mismatch; missing_in_actual={missing_in_actual}, "
                f"missing_in_expected={missing_in_expected}"
            )
            return
        for k in sorted(a.keys()):
            compare_values(a[k], b[k], f"{path}.{k}", errors)
        return

    if isinstance(a, list):
        if len(a) != len(b):
            errors.append(f"{path}: list length mismatch {len(a)} != {len(b)}")
            return
        for i, (av, bv) in enumerate(zip(a, b)):
            compare_values(av, bv, f"{path}[{i}]", errors)
        return

    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=FLOAT_TOL):
            errors.append(f"{path}: float mismatch {a} != {b} (tol={FLOAT_TOL})")
        return

    if a != b:
        errors.append(f"{path}: value mismatch {a!r} != {b!r}")


def compare_snapshots(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    compare_values(actual, expected, "root", errors)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["capture", "compare"], required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--requests-file", default=str(DEFAULT_REQUESTS))
    parser.add_argument("--snapshots-file", default=str(DEFAULT_SNAPSHOTS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requests_file = Path(args.requests_file)
    snapshots_file = Path(args.snapshots_file)

    try:
        actual = capture(args.base_url, requests_file)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    if args.mode == "capture":
        save_json(snapshots_file, actual)
        print(f"[ok] snapshots captured to {snapshots_file}")
        return 0

    if not snapshots_file.exists():
        print(f"[error] snapshots file not found: {snapshots_file}", file=sys.stderr)
        return 2

    expected = load_json(snapshots_file)
    errors = compare_snapshots(actual, expected)
    if errors:
        print("[fail] snapshot parity mismatch:")
        for err in errors[:200]:
            print(f"  - {err}")
        if len(errors) > 200:
            print(f"  ... and {len(errors) - 200} more differences")
        return 1

    print("[ok] snapshot parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
