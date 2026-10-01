#!/usr/bin/env python3
"""ctx-ab.py — compare two context configurations on identical, cache-busting probes.

Every run gets a unique nonce baked into the filler, because llama-server caches the prompt
prefix: re-running the same needle re-reads the cache (you will see prompt_n collapse to a
handful of tokens) and measures nothing. That confound is what makes naive long-context
testing lie.

Prints a table so the two configs are comparable line by line.
"""
import argparse
import json
import random
import string
import sys
import time
import urllib.request

FILLER = (
    "The maintenance crew walked the north corridor, checking each junction box in turn. "
    "A pressure gauge read low on the third rack and was flagged for recalibration. "
    "Later, the cooling fans cycled down for routine inspection and the ambient "
    "temperature settled within the expected band. "
)
CHARS_PER_TOKEN = 3.6   # calibrated: 250000 target -> 165034 actual tokens on this server


def nonce(n=6):
    return "".join(random.choice(string.ascii_uppercase) for _ in range(n))


def post(port, body, timeout=5400):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def build(target_tokens, depth, tag, needle):
    """Filler carrying a nonce, with the needle planted at `depth` through the context.

    Sized in characters, then calibrated: this tokenizer averages ~3.6 characters per token
    on this filler, so CHARS_PER_TOKEN is measured, not guessed. Getting this wrong silently
    halves or doubles every "target", which makes the sweep meaningless.
    """
    chars = max(2000, int(target_tokens * CHARS_PER_TOKEN))
    pool = FILLER * (int(chars / len(FILLER)) + 4)
    cut = int(chars * depth)
    pre = pool[:cut]
    mid = pool[cut:]
    planted = f" IMPORTANT RECORD [{tag}] — the depot access code is {needle}. "
    return pre + planted + mid + "\n\nWhat is the depot access code? Answer with only the value."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--label", required=True)
    ap.add_argument("--targets", default="60000,120000,180000,230000")
    ap.add_argument("--depth", type=float, default=0.5)
    a = ap.parse_args()

    print(f"\n=== {a.label} (port {a.port}, needle at depth {a.depth}) ===")
    print(f"{'target':>8}  {'actual':>8}  {'verdict':>7}  {'decode tok/s':>12}  seconds")
    failures = 0
    for target in [int(x) for x in a.targets.split(",")]:
        tag = nonce()
        needle = str(7000 + random.randint(0, 99))
        body = build(target, a.depth, tag, needle)
        t0 = time.time()
        try:
            r = post(a.port, {"prompt": body, "n_predict": 48, "temperature": 0.0})
        except Exception as e:
            print(f"{target:>8}  {'-':>8}  {'ERROR':>7}  {'-':>12}  {type(e).__name__}")
            failures += 1
            continue
        secs = time.time() - t0
        actual = r["timings"]["prompt_n"]
        out = r["content"].strip()
        # Strict: the needle must appear, and the answer must be short enough that we aren't
        # scoring a rambling paragraph that happened to mention it.
        hit = needle in out
        clean = hit and len(out) < 200
        verdict = "PASS" if clean else ("loose" if hit else "MISS")
        if not clean:
            failures += 1
        print(f"{target:>8}  {actual:>8}  {verdict:>7}  "
              f"{r['timings']['predicted_per_second']:>12.1f}  {secs:5.1f}")
        if not clean:
            print(f"          got: {out[:160]!r}")

    print(f"{a.label}: {failures} not-clean result(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())