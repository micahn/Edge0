#!/usr/bin/env bash
# bench-server.sh — start a server with a given arg set, measure prefill+decode, tear down.
#
# Usage: bench-server.sh "<label>" <extra llama-server args...>
# Env: EDGE0_8B_GGUF, EDGE0_8B_LORA, EDGE0_CTX, EDGE0_PORT (default 8099)
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

LABEL="$1"; shift
BIN="${EDGE0_BIN_DIR:-$ROOT/wt/win/build-vk/bin}"
MODEL="${EDGE0_8B_GGUF:-$ROOT/models/edge0-8b-gguf/edge0-8b.gguf}"
LORA="${EDGE0_8B_LORA:-$ROOT/models/edge0-8b/lora_edge0_8b-gguf.gguf}"
CTX="${EDGE0_CTX:-8192}"
PORT="${EDGE0_PORT:-8099}"
LOG="/tmp/bench-${PORT}.log"

pkill -f "llama-server -m .*--port ${PORT}" 2>/dev/null; sleep 2

args=(-m "$MODEL" -c "$CTX" --port "$PORT" --no-warmup)
[ -f "$LORA" ] && args+=(--lora "$LORA")
args+=("$@")

setsid "$BIN/llama-server" "${args[@]}" > "$LOG" 2>&1 < /dev/null &
SRV=$!

# wait for readiness
for _ in $(seq 1 90); do
  grep -q "listening on" "$LOG" && break
  grep -q "exiting due to" "$LOG" && { echo "$LABEL: FAILED TO START"; tail -5 "$LOG"; exit 1; }
  sleep 1
done
grep -q "listening on" "$LOG" || { echo "$LABEL: TIMEOUT"; tail -5 "$LOG"; kill $SRV 2>/dev/null; exit 1; }

PID=$(pgrep -f "llama-server -m .*--port ${PORT}" | head -1)
LOAD=$(grep -oE "load_tensors: offload [0-9]+/[0-9]+ layers" "$LOG" | tail -1)

python3 - "$PORT" "$LABEL" "$LOAD" "$PID" <<'PY'
import json, sys, urllib.request, statistics
port, label, load, pid = sys.argv[1:5]
def run(prompt_tok_target, n_predict=96):
    body = json.dumps({"prompt": ("Describe streaming expert offload in detail. " * prompt_tok_target),
                       "n_predict": n_predict, "temperature": 0.0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})))
    t = r["timings"]
    return t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"]

short = run(10)          # ~80 tok prompt
long_ = run(160)         # ~1.3k tok prompt
try:
    rss = int([l for l in open(f"/proc/{pid}/status") if l.startswith("VmHWM")][0].split()[1])
    rssmb = rss // 1024
except Exception:
    rssmb = -1
print(f"{label:<34} short: prefill {short[1]:7.0f}  decode {short[2]:6.1f} tok/s | "
      f"long({long_[0]}tok): prefill {long_[1]:7.0f}  decode {long_[2]:6.1f} tok/s | "
      f"peakRSS {rssmb:5d}MB {load}")
PY

kill $SRV 2>/dev/null; sleep 3
pkill -f "llama-server -m .*--port ${PORT}" 2>/dev/null
exit 0