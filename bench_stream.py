#!/usr/bin/env python3
"""Benchmark streaming précis : TTFT + débit de décodage soutenu.

Token count via usage.completion_tokens (stream_options.include_usage=true),
donc indépendant du nombre de caractères du flux.

Configuration par variables d'environnement (défauts entre parenthèses) :
  SGLANG_BASE_URL (http://127.0.0.1:1234/v1/chat/completions)
  SGLANG_MODEL    (qwen3.8-27b-int8-w8a16)
"""
import json, os, time, urllib.request, statistics

BASE = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:1234/v1/chat/completions")
MODEL = os.environ.get("SGLANG_MODEL", "qwen3.8-27b-int8-w8a16")


def stream_call(prompt, max_tokens, thinking=False, temperature=0.0):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if "usage" in d and d["usage"] is not None:
                completion_tokens = d["usage"].get("completion_tokens", 0)
                prompt_tokens = d["usage"].get("prompt_tokens", 0)
            choices = d.get("choices") or []
            if choices:
                ch = choices[0].get("delta", {})
                text = ch.get("content") or ch.get("reasoning_content")
                if text and ttft is None:
                    ttft = time.time() - t0
    total = time.time() - t0
    decode_t = total - ttft if ttft is not None else 0
    return ttft, total, prompt_tokens, completion_tokens, decode_t


def run(prompt, max_tokens, thinking, reps=2):
    ttfts, tpss = [], []
    for _ in range(reps):
        ttft, total, pt, ct, decode_t = stream_call(prompt, max_tokens, thinking)
        tps = ct / decode_t if decode_t > 0 else 0
        ttfts.append(ttft or 0)
        tpss.append(tps)
    return statistics.median(ttfts), statistics.median(tpss)


def main():
    print("Warmup...")
    stream_call("Say OK.", 16, False)
    print("OK\n")

    probes = [
        ("essai (prose, 1024 tok)", "Write a detailed technical essay on the history of computing, from Babbage to modern GPUs.", 1024, False),
        ("code (LRUCache, 1024 tok)", "Write a complete Python LRUCache class with O(1) get/put using OrderedDict, plus unit tests and a short usage example.", 1024, False),
        ("essai + thinking (800 tok)", "Write a detailed technical essay on the history of computing, from Babbage to modern GPUs.", 800, True),
    ]

    print(f"{'sonde':32s} {'TTFT':>7s} {'decode':>9s}")
    print("-" * 52)
    for label, prompt, mt, th in probes:
        ttft, tps = run(prompt, mt, th)
        print(f"{label:32s} {ttft:6.2f}s {tps:7.2f} tok/s")


if __name__ == "__main__":
    main()
