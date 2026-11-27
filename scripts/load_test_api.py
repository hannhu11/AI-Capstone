"""
FastAPI Concurrent Load & Stress Test (Task 3 Giai doan 3.2)
==========================================================
Measures concurrent users (10, 25, 50, 100), RPS, Avg/p50/p95 Latency (ms),
and Failure Rate (%) against the live FastAPI ASGI server endpoints.
"""
import argparse
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
import urllib.request

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_load_level(base_url: str, concurrency: int, total_requests: int):
    latencies_ms = []
    errors = 0
    lock = threading.Lock()
    endpoints = ["/api/metrics", "/api/status", "/api/tracking/workers", "/api/benchmark/summary"]

    def worker(idx: int):
        nonlocal errors
        url = base_url + endpoints[idx % len(endpoints)]
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                _ = resp.read()
                code = resp.getcode()
            dt = (time.perf_counter() - t0) * 1000.0
            with lock:
                if code == 200:
                    latencies_ms.append(dt)
                else:
                    errors += 1
        except Exception:
            with lock:
                errors += 1

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker, i) for i in range(total_requests)]
        for f in futs:
            f.result()
    wall_s = time.perf_counter() - t_start

    arr = np.array(latencies_ms, dtype=np.float64) if latencies_ms else np.array([0.0])
    rps = total_requests / wall_s
    err_pct = (errors / total_requests) * 100.0
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_s": round(wall_s, 3),
        "rps": round(rps, 1),
        "avg_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "errors": int(errors),
        "error_pct": round(err_pct, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default=None, help="Target an already-running external gateway (e.g. http://127.0.0.1:8080)")
    parser.add_argument("--out", type=str, default=str(ROOT / "Output" / "load_test_results.json"), help="Output JSON file path")
    args = parser.parse_args()

    server = None
    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        from src.web.app import app

        port = 18088
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{port}"

    # Wait for server ready
    for _ in range(60):
        try:
            with urllib.request.urlopen(base_url + "/api/status", timeout=2) as r:
                if r.getcode() == 200:
                    break
        except Exception:
            time.sleep(0.5)

    # Warmup
    _ = run_load_level(base_url, concurrency=4, total_requests=20)

    results = []
    for c, n_req in [(10, 200), (25, 300), (50, 400), (100, 500)]:
        rec = run_load_level(base_url, concurrency=c, total_requests=n_req)
        results.append(rec)
        print(f"[LOAD C={c}] {rec}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_path}", flush=True)
    if server is not None:
        server.should_exit = True


if __name__ == "__main__":
    main()
