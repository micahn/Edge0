#!/usr/bin/env python3
"""set-ctx-length.py — raise (or restore) the context_length recorded in an edge0 GGUF.

Why this exists
---------------
llama-server hard-caps every slot at the model's *training* context:

    tools/server/server-context.cpp:4054
        return std::min(res, llama_model_n_ctx_train(model_tgt));

`n_ctx_train` comes from the GGUF's `<arch>.context_length` KV. So `-c 262144` is silently
clamped back to 131072, and the RoPE scaling flags (`--rope-scaling`, `--rope-scale`,
`--yarn-*`) have nothing to act on — the context never grows, so there is nothing to scale.
The server does warn "possible training context overflow", then serves 131072 anyway.

Editing that one metadata field is what makes extrapolation reachable. It is a metadata claim,
not a capability change: the model's arithmetic is unchanged, and whether it actually holds up
past its trained length is a question you have to measure (see linux/needle.py).

The value is a single u32 inside the KV block, so patching it in place keeps every offset in
the file valid — no repack, no offset table to rewrite.

    set-ctx-length.py models/edge0-8b.gguf              # report
    set-ctx-length.py models/edge0-8b.gguf 262144       # raise
    set-ctx-length.py models/edge0-8b.gguf 131072       # restore

Requires the file to be mmapped by no running server, and to be re-readable afterwards.
"""
import os
import struct
import sys

KEY = b"bailingmoe3.context_length"
U32 = 4


def find_kv(path, needle=KEY):
    """Locate the value slot for `<arch>.context_length` by walking the KV block.

    Searches for the key bytes rather than parsing every field: the GGUF KV block is
    self-delimiting but hand-walking it desyncs on array element types, and a byte search for
    a unique key is both simpler and verifiable — we assert the type and the current value
    before writing anything.
    """
    with open(path, "rb") as f:
        blob_start, blob_end = 0, os.path.getsize(path)
        chunk = 1 << 22
        pos = 0
        prev = b""
        while pos < blob_end:
            f.seek(pos)
            buf = f.read(chunk)
            if not buf:
                break
            hay = prev + buf
            at = 0
            while True:
                i = hay.find(needle, at)
                if i < 0:
                    break
                # absolute offset of the key bytes within the file
                key_abs = pos - len(prev) + i
                _verify(f, key_abs)
                return key_abs + len(needle) + 4  # +4 skips the u32 type tag
                at = i + 1
            prev = buf[-len(needle):]
            pos += len(buf)
    raise SystemExit(f"{needle.decode()} not found in {path}")


def _verify(f, key_abs):
    """Assert the bytes around the key really are one u32 field before anyone writes to it."""
    f.seek(key_abs - 8)
    klen = struct.unpack("<Q", f.read(8))[0]
    if klen != len(KEY):
        raise SystemExit(f"length prefix {klen} != {len(KEY)} at {key_abs - 8}: not our field")
    f.seek(key_abs + len(KEY))
    typ = struct.unpack("<I", f.read(4))[0]
    if typ != U32:
        raise SystemExit(f"field type {typ} != {U32}: refusing to patch")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    off = find_kv(path)

    with open(path, "rb") as f:
        f.seek(off)
        cur = struct.unpack("<I", f.read(4))[0]

    if len(sys.argv) == 2:
        print(f"{path}\n  {KEY.decode()} = {cur} (at byte {off})")
        return 0

    want = int(sys.argv[2])
    if not (0 < want <= 2 ** 31 - 1):
        raise SystemExit(f"refusing implausible context_length: {want}")

    with open(path, "r+b") as f:
        f.seek(off)
        f.write(struct.pack("<I", want))
        f.flush()
        os.fsync(f.fileno())

    print(f"{path}\n  {KEY.decode()}: {cur} -> {want}")
    if want > cur:
        print(f"  NOTE: past the model's trained {cur}. Pair with RoPE scaling, e.g.")
        print(f"        --rope-scaling linear --rope-scale {want / cur:g}")
        print("  and verify with linux/needle.py before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)