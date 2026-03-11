# Snapshot Parity Harness

This folder provides strict JSON parity checks for backend endpoints after refactors.

## Files

- `requests.json`: fixed request cases.
- `snapshots.json`: golden response snapshots (generated in capture mode).
- `run_snapshot_parity.py`: capture/compare runner.

## Usage

Start backend first (`http://127.0.0.1:8000` by default).

Capture baseline snapshots:

```powershell
python backend/tests/parity/run_snapshot_parity.py --mode capture
```

Compare current backend against snapshots:

```powershell
python backend/tests/parity/run_snapshot_parity.py --mode compare
```

Optional args:

- `--base-url`
- `--requests-file`
- `--snapshots-file`

## Comparison Rules

- Exact dict key match.
- Exact list length and ordering.
- Float tolerance `1e-9`.
- Datetime strings compared exactly.
