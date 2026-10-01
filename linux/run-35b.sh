#!/usr/bin/env bash
# run-35b.sh — start the Linux/Vulkan edge0-35B server.
#
# SAME RAM CAVEAT AS THE UPSTREAM README: a ~21.75 GB GGUF against 31 GB of RAM. Expect the
# OS to page and decode to be slower than the 8B tier, not faster, despite the larger model.
# The 35B is not faster here — it is a different model. Benchmarks in linux/README.md.
#
# The prerouter head IS wired for this tier (upstream hardcodes its shapes to 35B: E=256,
# K=4, owners 6..38), so E0_PREROUTER is enabled here and dormant on 8B.
#
# Default port is 8082 so the two tiers don't collide — but do NOT run them together:
# 5 GB + 21.75 GB of weights against 31 GB of RAM will page hard.
#
# CTX defaults to 131072, the checkpoint native window. Do NOT lower it below 32768 while
# driving this through opencode: its system prompt plus ~59 MCP tool definitions exceeds 8k
# tokens, so a smaller window makes every session try to compact before its first token and
# fail with "The compaction request cannot be reduced further".
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

BIN="${EDGE0_BIN_DIR:-$ROOT/wt/win/build-vk/bin}"
MODEL="${EDGE0_35B_GGUF:-$ROOT/models/edge0-35b-gguf/edge0-35b.gguf}"
LORA="${EDGE0_35B_LORA:-$ROOT/models/edge0-35b/lora_edge0_35b-gguf.gguf}"
POOL_MB="${EDGE0_POOL_MB:-512}"
PORT="${EDGE0_35B_PORT:-8082}"
CTX="${EDGE0_35B_CTX:-131072}"

[ -x "$BIN/llama-server" ] || { echo "engine not built: bash linux/scripts/vendor-build.sh" >&2; exit 1; }
[ -f "$MODEL" ]            || { echo "GGUF not found: $MODEL (run windows/tools/repack_r3.py)" >&2; exit 1; }

args=(-m "$MODEL" -ngl 99 -fa on -c "$CTX" -np 1 --pool-mb "$POOL_MB" --port "$PORT")

# The LoRA adapter costs ~20% decode throughput. Kept by default; EDGE0_NO_LORA=1 to A/B.
if [ -n "${EDGE0_NO_LORA:-}" ]; then
  echo "note: EDGE0_NO_LORA set — running the plain int4 base (faster, lower quality)" >&2
elif [ -f "$LORA" ]; then
  args+=(--lora "$LORA")
else
  echo "note: no LoRA adapter at $LORA — running the plain base" >&2
fi

# Prerouter head: shapes are the 35B ones upstream, so this tier is the one that can use it.
export E0_POOL="${E0_POOL:-sticky}"       # route expert reads through the L1 pool
export E0_PREROUTER="${E0_PREROUTER:-$ROOT/models/edge0-35b/prerouter_edge0_35b.safetensors}"

echo "note: ~21.75 GB model against 31 GB RAM — paging-bound, decode will be slower than 8B" >&2
exec "$BIN/llama-server" "${args[@]}"