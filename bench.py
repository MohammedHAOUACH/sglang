#!/usr/bin/env python3
"""Benchmark simple du serveur SGLang Qwen3.8-27B-INT8-W8A16-MTP.

Mesure :
  - débit de décodage net (tok/s) via la méthode delta 2 appels (court/long),
  - TTFT (time-to-first-token) sur un prompt long,
  - mode thinking vs non-thinking.
"""
import json, time, urllib.request, urllib.error, uuid, sys

BASE = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen3.8-27b-int8-w8a16"


def call(prompt, max_tokens, thinking=None, temperature=None, stream=False):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    if temperature is not None:
        body["temperature"] = temperature
    if stream:
        body["stream"] = True
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        dt = time.time() - t0
        usage = d.get("usage", {})
        return dt, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    # streaming : on lit jusqu'au premier token
    first = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if first is None and b"data:" in line and b"delta" in line:
                first = time.time()
                break
    return (first or time.time()) - t0, 0, 0


def net_decode(prompt, thinking, short=60, long=600):
    d1, p1, c1 = call(prompt, short, thinking)
    d2, p2, c2 = call(prompt, long, thinking)
    tps = (c2 - c1) / (d2 - d1) if (d2 - d1) > 0 else float("nan")
    return tps, c1, d1, c2, d2


def main():
    print(f"Benchmark contre {BASE}\n")
    # warmup (1er appel = fin de capture / warmup cache)
    print("Warmup...")
    call("Say OK.", 16, False)
    print("Warmup terminé.\n")

    probes = [
        ("code (LRUCache)", "Write a Python class LRUCache with O(1) get and put using OrderedDict, plus a small test.", False),
        ("essai (prose longue)", "Write a detailed technical essay on the history of computing, from Babbage to GPUs.", False),
        ("chat court (hash map)", "What is a hash map and why is it useful? Answer concisely.", False),
        ("code + thinking", "Write a Python class LRUCache with O(1) get and put using OrderedDict, plus a small test.", True),
    ]

    print(f"{'sonde':32s} {'net decode':>10s}   {'c1':>5s} {'d1':>6s}  {'c2':>5s} {'d2':>6s}")
    print("-" * 82)
    results = {}
    for name, prompt, th in probes:
        tps, c1, d1, c2, d2 = net_decode(prompt, th)
        results[name] = tps
        print(f"{name:32s} {tps:9.2f} tok/s   {c1:5d} {d1:6.2f}s  {c2:5d} {d2:6.2f}s")

    # TTFT sur prompt long (~16k tokens), non-thinking, streaming
    print("\nTTFT sur prompt long (~16k tokens)...")
    para = ("The quick brown fox jumps over the lazy dog while the sun sets "
            "over the mountains and rivers flow gently through the valley. ") * 640
    prompt = f"Session {uuid.uuid4()}. Summarize in one sentence:\n\n{para}"
    ttft, _, _ = call(prompt, 32, thinking=False, stream=True)
    nwords = len(prompt.split())
    print(f"TTFT        : {ttft:6.2f}s   (prefill ~{nwords/ttft:7.0f} tok/s)")

    print("\n=== Récapitulatif tok/s (décodage net) ===")
    for name, tps in results.items():
        print(f"  {name:32s} {tps:6.2f} tok/s")


if __name__ == "__main__":
    main()
