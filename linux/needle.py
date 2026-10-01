#!/usr/bin/env python3
"""needle.py — measure context quality for the edge0-8B server.

Three probes, cheapest first, because they fail in different ways:

  arith     "17 * 3" -> 51. Proves the graph runs and the head is coherent.
  recall    A fact planted at a known depth in filler prose, asked at the end. Proves the
            content actually survived in the KV cache. This is the probe that RoPE scaling
            breaks first and the one that matters.
  tail      The same fact planted at the very end. Controls for "can it answer at all"
            versus "can it still reach the start of the context".

Usage:
  needle.py --port 8081 [--depth 0.05] [--fill-tokens 40000]

Reports per-depth recall so you can see the curve collapse rather than just a single
pass/fail. Exits non-zero if any probe misses, so it can gate a change.
"""
import argparse
import json
import sys
import time
import urllib.request

FILLER = (
    "The maintenance crew walked the north corridor, checking each junction box in turn. "
    "A pressure gauge read low on the third rack and was flagged for recalibration. "
    "Later, the cooling fans cycled down for routine inspection and the ambient "
    "temperature settled within the expected band. "
)


def post(port, body, timeout=3600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def arith(port):
    r = post(port, {"prompt": "What is 17 times 3? Reply with only the number.",
                    "n_predict": 48, "temperature": 0.0})
    got = r["content"].strip()
    ok = "51" in got
    return ok, f"{got[:60]!r}"


def needle_facts(i):
    """Distinctive, unambiguous facts. Numbers alone invite lucky guesses."""
    return {
        "code": f"the depot access code is {7000 + i}",
        "site": f"the relay station is named Kestrel {i}",
        "owner": f"the duty supervisor is Priya {i}",
    }


def recall(port, depth, fill_tokens, n_predict=64):
    """Plant one fact at `depth` of the way through a filler context, then ask."""
    words = fill_tokens // 9                     # FILLER is ~9 words per sentence
    total = max(words, 16)
    cut = max(1, min(total - 1, int(total * depth)))

    fact = needle_facts(1)
    key, value = next(iter(fact.items()))
    question = {
        "code": "What is the depot access code?",
        "site": "What is the relay station named?",
        "owner": "Who is the duty supervisor?",
    }[key]

    pre = (FILLER * ((cut // 4) + 1))[: cut * 6]
    mid = (FILLER * ((cut // 4) + 1))[cut * 6:]
    planted = f" IMPORTANT RECORD — {value}. "
    body = pre + planted + mid + "\n\n" + question + " Answer with only the value."

    t0 = time.time()
    r = post(port, {"prompt": body, "n_predict": n_predict, "temperature": 0.0})
    got = r["content"].strip()
    # The value's tail (e.g. "7001" from "the depot access code is 7001") must appear.
    needle = value.split()[-1]
    ok = needle in got
    return ok, f"want {needle!r} got {got[:60]!r}", r["timings"]["prompt_n"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--depths", default="0.02,0.25,0.6,0.9")
    ap.add_argument("--fill-tokens", type=int, default=40000)
    a = ap.parse_args()

    ok, got = arith(a.port)
    print(f"[arith] {'PASS' if ok else 'FAIL'}  {got}")
    failures = 0 if ok else 1

    print(f"\n[recall] fill≈{a.fill_tokens} tok")
    for d in [float(x) for x in a.depths.split(",")]:
        ok, detail, ntok, secs = recall(a.port, d, a.fill_tokens)
        failures += 0 if ok else 1
        print(f"  depth {d:<5} {a.fill_tokens:>6} tok  {'PASS' if ok else 'FAIL'}  "
              f"prompt={ntok:<6} {secs:5.1f}s  {detail}")

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())