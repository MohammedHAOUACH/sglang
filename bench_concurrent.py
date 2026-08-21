#!/usr/bin/env python3
"""Test de charge concurrent : débit agrégé vs débit par requête.

Configuration par variables d'environnement (défauts entre parenthèses) :
  SGLANG_BASE_URL (http://127.0.0.1:1234/v1/chat/completions)
  SGLANG_MODEL    (qwen3.8-27b-int8-w8a16)
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:1234/v1/chat/completions")
MODEL = os.environ.get("SGLANG_MODEL", "qwen3.8-27b-int8-w8a16")
MAX_TOKENS = 128

PROMPT = (
    "Write a short but detailed explanation of how continuous batching "
    "works in an LLM inference server. Then add extra detail about the "
    "scheduler, KV cache, and chunked prefill until you are clearly done."
)


def run_one(idx):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": f"Worker {idx}."},
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        BASE,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
    total = time.perf_counter() - t0
    c = completion_tokens or chunks
    gen_s = max(total - (ttft or 0), 1e-6)
    return {
        "idx": idx,
        "tokens": c,
        "total_s": total,
        "ttft_s": ttft,
        "decode_tps": c / gen_s,
    }


def bench(n):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(run_one, range(n)))
    wall = time.perf_counter() - t0
    total_tokens = sum(r["tokens"] for r in results)
    agg_tps = total_tokens / wall
    avg_lat = sum(r["total_s"] for r in results) / n
    avg_tps = sum(r["decode_tps"] for r in results) / n
    print(f"--- {n} concurrent request(s) ---")
    print(f"  wall time:           {wall:8.2f} s")
    print(f"  total tokens:        {total_tokens:8d}")
    print(f"  AGGREGATE throughput:{agg_tps:8.1f} tok/s")
    print(f"  avg per-request:     {avg_tps:8.1f} tok/s   (avg latency {avg_lat:.2f} s)")
    print()
    return {"n": n, "agg_tps": agg_tps, "avg_tps": avg_tps, "wall": wall}


def main():
    print(f"Warm-up ({MAX_TOKENS} tokens) ...", flush=True)
    run_one(0)
    print("WARM-UP DONE\n", flush=True)
    rows = []
    for n in [1, 2, 4, 8]:
        rows.append(bench(n))
    print("=== SUMMARY ===")
    print(f"{'concurrent':>10} {'aggregate':>10} {'per-request':>12}")
    for r in rows:
        print(f"{r['n']:>10} {r['agg_tps']:>9.1f}/s {r['avg_tps']:>11.1f}/s")


if __name__ == "__main__":
    main()
