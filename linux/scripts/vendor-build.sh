#!/usr/bin/env bash
# linux/scripts/vendor-build.sh — llama engine assembly for Linux (monorepo layout).
#
# The pristine upstream tree (../vendor/llama.cpp, pinned commit) is NEVER patched in place:
# this script replays the patch bands into an isolated build worktree (../wt/win — same
# worktree the Windows path uses, since the bands are identical), copies the edge0 serving
# pieces from linux/serve/ into src/edge0 (patch #3's CMake glob compiles them into the llama
# target), then configures and builds with Vulkan.
#
# This is the POSIX counterpart of windows/scripts/vendor-build.ps1. Same depot resolution:
#   A. $EDGE0_DEPOT                explicit override
#   B. parent directory containing vendor.llama.pin   monorepo checkout (the normal path)
# Set EDGE0_LLAMA_URL to clone from a mirror.
#
# Usage: scripts/vendor-build.sh [-d build-vk] [--clean] [--assemble-only]
set -euo pipefail

BUILD_DIR="build-vk"
CLEAN=0
ASSEMBLE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    -d) BUILD_DIR="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --assemble-only) ASSEMBLE_ONLY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Depot = the tree carrying vendor.llama.pin: EDGE0_DEPOT, else the nearest ancestor of this
# script that has it (the monorepo root, two levels up from linux/scripts/).
DEPOT="${EDGE0_DEPOT:-}"
if [ -z "$DEPOT" ]; then
  d=$(dirname "$SELF")
  while [ "$d" != "/" ]; do
    if [ -f "$d/vendor.llama.pin" ]; then DEPOT="$d"; break; fi
    d=$(dirname "$d")
  done
fi
[ -n "$DEPOT" ] || { echo "no depot: set EDGE0_DEPOT to a tree carrying vendor.llama.pin, or run inside a full monorepo clone" >&2; exit 1; }
SERVE="$DEPOT/linux/serve"
[ -d "$SERVE" ] || { echo "serving pieces missing at $SERVE" >&2; exit 1; }

PIN=$(awk '{print $1}' "$DEPOT/vendor.llama.pin")
VENDOR="$DEPOT/vendor/llama.cpp"
WT="$DEPOT/wt/win"
PDIR="$DEPOT/patches/llama.cpp"
[ -d "$PDIR/common" ] || { echo "patch bands missing at $PDIR/common" >&2; exit 1; }
[ -d "$PDIR/windows" ] || { echo "windows patch band missing at $PDIR/windows" >&2; exit 1; }

# Materialize the pinned tree when absent (submodule-free supply).
if [ ! -d "$VENDOR/.git" ]; then
  echo "[0/4] materializing vendor: clone llama.cpp at pinned $PIN -> $VENDOR"
  LLAMA_URL="${EDGE0_LLAMA_URL:-https://github.com/ggml-org/llama.cpp}"
  mkdir -p "$(dirname "$VENDOR")"
  git clone --filter=blob:none "$LLAMA_URL" "$VENDOR"
  git -C "$VENDOR" checkout --detach "$PIN"
fi

echo "[1/4] checking pristine vendor tree (must sit exactly at the pin, zero local changes)"
head=$(git -C "$VENDOR" rev-parse HEAD)
case "$head" in
  "$PIN"*) ;;
  *) echo "vendor HEAD=$head is not the pinned $PIN — restore with: git -C $VENDOR checkout --detach $PIN" >&2; exit 1 ;;
esac
[ -z "$(git -C "$VENDOR" status --porcelain)" ] || { echo "vendor worktree is dirty — the pristine source of truth tolerates no local edits" >&2; exit 1; }

echo "[2/4] (re)creating build worktree $WT and replaying patch bands"
if [ "$CLEAN" = 1 ] && [ -e "$WT" ]; then
  git -C "$VENDOR" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
  git -C "$VENDOR" worktree prune
fi
if [ ! -e "$WT" ]; then git -C "$VENDOR" worktree add --detach "$WT" "$PIN" >/dev/null; fi
# Unconditional reset before replay = idempotent
git -C "$WT" reset --hard "$PIN" >/dev/null
git -C "$WT" clean -fdq src/edge0 tools/llama-bench
for band in common windows; do
  for p in "$PDIR/$band"/*.patch; do
    echo "  am $band/$(basename "$p")"
    if ! git -C "$WT" am --3way "$p"; then
      echo "git am failed at $p (upstream drift or band conflict — see patches/llama.cpp/README.md)" >&2
      exit 1
    fi
  done
done

echo "[3/4] copying linux/serve pieces -> src/edge0/ (compiled in via patch #3 glob)"
mkdir -p "$WT/src/edge0"
cp -f "$SERVE"/*.cc "$SERVE"/*.h "$WT/src/edge0/"

if [ "$ASSEMBLE_ONLY" = 1 ]; then
  echo "OK(assemble-only): worktree=$WT — patches replayed, serving pieces copied"
  echo "  tree hash: $(git -C "$WT" rev-parse HEAD^{tree})"
  exit 0
fi

echo "[4/4] configure + build (Release/Vulkan)"
bd="$WT/$BUILD_DIR"
JOBS="${JOBS:-$(nproc)}"
cmake -S "$WT" -B "$bd" \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$bd" -j "$JOBS"
echo "OK: $bd/bin/llama-server"