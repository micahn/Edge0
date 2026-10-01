#!/usr/bin/env bash
# fetch-tier.sh <hf-repo> <dest-dir> <rate>  — resumable, rate-limited per-file fetch of an
# edge0 model repo. Skips media/demo assets. Written for the Linux port (the repo's own
# helper, python/scripts/fetch_models.py, only targets macOS tooling).
set -uo pipefail

REPO="$1"; DEST="$2"; RATE="$3"
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DEST"

files=$(curl -sf "https://huggingface.co/api/models/${REPO}" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("\n".join(f["rfilename"] for f in d["siblings"]))')

for f in $files; do
  case "$f" in
    *.jpg|*.mp4|*.png|.gitattributes) continue ;;
  esac
  out="${DEST}/${f}"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then echo "skip   $f"; continue; fi
  echo "get    $f"
  curl -fL --retry 5 --retry-delay 3 --continue-at - \
       --limit-rate "${RATE}" \
       -o "${out}.part" "${BASE}/${f}" \
    && mv -f "${out}.part" "$out" \
    || { echo "FAILED $f" >&2; rm -f "${out}.part"; }
done
echo "done   ${REPO} -> ${DEST}"