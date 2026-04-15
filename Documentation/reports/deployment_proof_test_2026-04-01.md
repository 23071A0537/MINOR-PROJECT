# Deployment Proof Test (2026-04-01)

## Scope

Validate two claims:

1. No real-time system exists.
2. No latency/throughput analysis exists.

## Test Environment

- OS: Windows
- Python: .venv (Python 3.11.9)
- Important fix applied before testing: downgraded NumPy to `<2` to match installed PyTorch binary compatibility.

## Test A: Real-time System Check

### Evidence from code

- `src/qids/inference/run_pipeline.py` triggers `src/qids/inference/live_inference.py` as a one-shot process and then exits.
- `src/qids/inference/live_inference.py` reads CSV files via `input_glob`, performs batch prediction, writes JSON artifacts, and exits.
- No server endpoint framework (FastAPI/Flask/Uvicorn/WebSocket/Kafka listener) was found in `src/**/*.py`.

### Runtime behavior observed

- Each inference invocation returned exit code `0` and terminated after artifact creation.
- No persistent daemon/service loop was started.

### Result

Claim 1 is confirmed: current implementation is batch file inference, not a continuous real-time deployment service.

## Test B: Latency and Throughput Check

### Existing project behavior

- No built-in latency/throughput measurement output is produced by `live_inference.py`.
- Therefore, explicit benchmark runs were executed to generate measurements.

### Benchmark method used

- Generated deterministic CSV input from Stage-2 test data at multiple sizes.
- Ran `src/qids/inference/live_inference.py` repeatedly.
- Used wall-clock timing around each full invocation.
- SHAP and LIME were disabled for baseline model-inference timing.

### Results (saved to artifacts/reports/live_inference_latency_throughput_test.json)

First sweep:

- 10 rows: 8.454598 s, 1.183 rows/s, 845.460 ms/row
- 25 rows: 8.281796 s, 3.019 rows/s, 331.272 ms/row
- 50 rows: 8.825996 s, 5.665 rows/s, 176.520 ms/row
- 100 rows: 8.062866 s, 12.403 rows/s, 80.629 ms/row
- 200 rows: 8.217027 s, 24.340 rows/s, 41.085 ms/row

Extended sweep:

- 200 rows: 6.802283 s, 29.402 rows/s, 34.011 ms/row
- 500 rows: 6.182928 s, 80.868 rows/s, 12.366 ms/row
- 1000 rows: 5.958203 s, 167.836 rows/s, 5.958 ms/row
- 2000 rows: 8.893277 s, 224.889 rows/s, 4.447 ms/row

Repeated 1000-row trials (5 runs):

- Runs (s): 11.452322, 9.534416, 6.949743, 7.215980, 6.398114
- Mean: 8.310115 s
- Std dev: 1.901361 s
- P95 (sample): 9.534416 s
- Mean throughput: 120.335 rows/s
- Mean latency: 8.310 ms/row

### Interpretation

- There is significant fixed startup overhead per invocation (model and artifact loading).
- Throughput improves with larger batch size because startup cost is amortized.
- Single-run response latency is not in real-time serving form because execution is offline batch invocation.

### Result

Claim 2 is confirmed for the current codebase state: there was no native latency/throughput analysis module; measurements had to be produced externally.

## Overall Verdict

Both concerns are valid for the current implementation:

1. No real-time serving system.
2. No built-in latency/throughput analysis.

This report adds objective test evidence for both points.
