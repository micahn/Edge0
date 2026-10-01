#!/usr/bin/env bash
# run-8b.sh — start the Linux/Vulkan edge0-8b server with the repo's load params.
#
# Mirrors the load contract from windows/README.md §1.5, minus the flags that have no
# Linux meaning (--mem-budget-mb is inert here; see linux/README.md).
#
# Flags here are the measured-best config for a single-user chat workload on
# Ryzen 7 5800X + RX 9070; see linux/README.md "Tuning" for the full matrix.
# Env overrides: EDGE0_PORT EDGE0_CTX EDGE0_NP EDGE0_POOL_MB EDGE0_NO_LORA=1
#
# CTX defaults to 131072, the checkpoint native window. Do NOT lower it below 32768 while
# driving this through opencode: its system prompt plus ~59 MCP tool definitions exceeds 8k
# tokens, so a smaller window makes every session try to compact before its first token and
# fail with "The compaction request cannot be reduced further".
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

BIN="${EDGE0_BIN_DIR:-$ROOT/wt/win/build-vk/bin}"
MODEL="${EDGE0_8B_GGUF:-$ROOT/models/edge0-8b-gguf/edge0-8b.gguf}"
LORA="${EDGE0_8B_LORA:-$ROOT/models/edge0-8b/lora_edge0_8b-gguf.gguf}"
POOL_MB="${EDGE0_POOL_MB:-512}"
PORT="${EDGE0_PORT:-8081}"
CTX="${EDGE0_CTX:-131072}"
NP="${EDGE0_NP:-1}"          # 1 slot: the server default (4, unified KV) costs ~15% on single-user chat

[ -x "$BIN/llama-server" ] || { echo "engine not built: bash linux/scripts/vendor-build.sh" >&2; exit 1; }
[ -f "$MODEL" ]            || { echo "GGUF not found: $MODEL (run windows/tools/repack_r3_8b.py)" >&2; exit 1; }

# -fa on: FlashAttention. ~2% on this box, but free.
args=(-m "$MODEL" -ngl 99 -fa on -c "$CTX" -np "$NP" --pool-mb "$POOL_MB" --port "$PORT")

# The LoRA adapter costs ~20% decode throughput (per-layer adapter math) and buys back most
# of the int4 quantization loss. On by default — drop it with EDGE0_NO_LORA=1 to A/B.
if [ -n "${EDGE0_NO_LORA:-}" ]; then
  echo "note: EDGE0_NO_LORA set — running the plain int4 base (faster, lower quality)" >&2
elif [ -f "$LORA" ]; then
  args+=(--lora "$LORA")
else
  echo "note: no LoRA adapter at $LORA — running the plain base" >&2
fi

export E0_POOL="${E0_POOL:-sticky}"       # route expert reads through the L1 pool

# The prerouter head is only wired for the 35B shapes upstream (E=256/K=4, owners 6..38).
# On 8B it parses but then fails the gate-weight check and stays dormant with a harmless
# "[pref-trace] gate missing/bad type" line, so don't enable it by default here.
if [ -n "${E0_PREROUTER:-}" ]; then export E0_PREROUTER; fi

exec "$BIN/llama-server" "${args[@]}"