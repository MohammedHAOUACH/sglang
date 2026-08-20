#!/usr/bin/env python3
"""Quick streaming speed test for the local sglang server."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:1234/v1"
MODEL = "qwen3.8-27b-int8-w8a16"


def stream(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    prompt_tokens = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=300) as r:
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
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                # reasoning models stream reasoning_content before content
                if delta.get("content") or delta.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_tokens += 1
    total = time.perf_counter() - t0
    return {
        "ttft_s": ttft,
        "total_s": total,
        "streamed_chunks": n_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def main():
    print("Warming up (prefill path) ...", flush=True)
    stream("Say hello.", max_tokens=8)
    print("WARM-UP DONE\n", flush=True)

    print("Running benchmark: 256-token decode ...", flush=True)
    res = stream(
        "Write a short but detailed summary of how continuous batching "
        "works in an LLM inference server, then keep going with extra "
        "detail until you are clearly done.",
        max_tokens=256,
    )
    c = res["completion_tokens"] or res["streamed_chunks"]
    gen_s = res["total_s"] - (res["ttft_s"] or 0)
    print("\n=== RESULT ===")
    print(f"TTFT (first token):      {res['ttft_s']*1000:8.1f} ms")
    print(f"Total time:              {res['total_s']:8.2f} s")
    print(f"Decode time (excl TTFT): {gen_s:8.2f} s")
    print(f"Completion tokens:       {c:8d}")
    print(f"Prompt tokens:           {res['prompt_tokens']}")
    if res["ttft_s"]:
        print(f"\nPrefill/TTFT rate:       {(res['prompt_tokens'] or 0)/res['ttft_s']:8.1f} tok/s (prompt/TTFT)")
    if gen_s > 0:
        print(f"DECODE throughput:       {c/gen_s:8.1f} tok/s")
    if res["total_s"] > 0:
        print(f"End-to-end (incl prompt):{c/res['total_s']:8.1f} tok/s")


if __name__ == "__main__":
    main()
