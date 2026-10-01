#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""repack_mlx_to_gguf.py — lossless-repack core shared by the edge0 MLX→GGUF converters.

This module was referenced by every converter in `windows/tools/` (repack_r3.py,
repack_r3_8b.py, lora_mlx_to_gguf.py, and convert_mlx_to_gguf.py shells out to them) but
was never committed upstream. It is reimplemented here against the API those callers use
and against the two format contracts they document:

  * safetensors: 8-byte little-endian header length, then JSON, then a flat data blob
    addressed by per-tensor [begin, end) byte offsets.
  * GGUF: "GGUF" magic, u32 version, u64 tensor_count, u64 kv_count, then the KV block,
    then the tensor-info block, then ALIGN-padded tensor data.

The one numerically subtle piece is the Q4_1 repack, documented in full at
`pack_q41_rows` below. The invariant every caller asserts is a dequantized-side equality
(mlx vs gguf), and that invariant is exact here — see the module-level note on why.
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

# ── GGML tensor-type enum (NOT the GGUF KV-type enum — they are different tables) ──
T_F32, T_F16, T_BF16, T_Q4_1 = 0, 1, 2, 3

# ── GGUF KV value types ──
KV = {
    "u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5,
    "f32": 6, "bool": 7, "str": 8, "arr": 9, "u64": 10, "i64": 11, "f64": 12,
}
_KV_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

ALIGN = 32          # general.alignment written by the callers
QK41 = 32          # GGML_TYPE_Q4_1 values per block
DT_Q41 = T_Q4_1
_Q41_BLOCK_BYTES = 20   # fp16 d + fp16 m + 16 packed-nibble bytes

# safetensors dtype → numpy
_ST_DT = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "BF16": np.uint16, "I64": np.int64, "I32": np.int32,
    "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}


def sha16(b):
    """16-hex-char digest. The converters' sentinel gates compare this across the two
    dequantization sides, so it must be stable and endian-explicit (raw bytes, no repr)."""
    return hashlib.sha256(b).hexdigest()[:16]


def bf16_to_f32(u):
    """uint16 bit pattern(s) → float32, widening bf16 (8 exp bits) into fp32."""
    u = np.asarray(u, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


def _f32_to_bf16_bits(x):
    """float32 → bf16 bit pattern, round-to-nearest-even (matches MLX / torch)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    # add 0x7FFF + lsb-of-kept-mantissa so the carry lands the tie correctly
    lsb = (u >> 16) & 1
    rounded = u + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_f32(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# ══════════════════════════════ safetensors ══════════════════════════════

class StIndex:
    """Read-only index over one or more safetensors shards.

    `.t[name]` is a 5-tuple `(data_start, data_end, nbytes, dtype, shape)` — the converters
    read index 3 for dtype and index 4 for shape. `.load(name)` materializes an ndarray.
    Handles are opened lazily and kept open for the lifetime of the index so that a
    multi-shard scan does not re-open per tensor.
    """

    def __init__(self, paths):
        self.paths = list(paths)
        self.t = {}
        self._fh = {}
        self._base = {}
        for path in self.paths:
            fh = open(path, "rb")
            self._fh[path] = fh
            hn = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hn))
            base = 8 + hn
            self._base[path] = base
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.t[k] = (a, b, b - a, v["dtype"], tuple(v["shape"]), path)

    def load(self, name):
        if name not in self.t:
            raise KeyError(f"safetensors tensor not found: {name}")
        a, b, _n, dt, shape, path = self.t[name]
        fh = self._fh[path]
        fh.seek(self._base[path] + a)
        raw = fh.read(b - a)
        if dt == "BF16":
            arr = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16))
        elif dt == "U32":
            arr = np.frombuffer(raw, dtype=np.uint32)
        elif dt == "U16":
            arr = np.frombuffer(raw, dtype=np.uint16)
        elif dt == "I64":
            arr = np.frombuffer(raw, dtype=np.int64)
        else:
            arr = np.frombuffer(raw, dtype=_ST_DT[dt])
        return arr.reshape(shape)

    def close(self):
        for fh in self._fh.values():
            try:
                fh.close()
            except Exception:
                pass
        self._fh.clear()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ══════════════════════════════ GGUF writer ══════════════════════════════

def _align_up(x, a):
    return (a - 1 + x) & ~(a - 1)


class GgufWriter:
    """Streaming GGUF writer.

    Two-phase by design, matching how every caller uses it:
      1. `kv*` metadata, then `register(name, type, dims, size)` per tensor, then `finalize()`
         — which writes the header and lays out the data section;
      2. `write(bytes)` + `align_pad()` per tensor in registration order, then `close()`.
    `dlen` tracks bytes emitted so far, which the callers use for progress lines.
    """

    def __init__(self, path, alignment=ALIGN):
        self.path = path
        self.align = alignment
        self.kv_pairs = []          # (key, vtype, raw value bytes)
        self.tensors = []           # (name, gtype, dims, size)
        self.dlen = 0
        self._f = open(path, "wb")
        self._data_off = 0
        self._final = False

    # ---- metadata ----
    @staticmethod
    def _str(s):
        b = s.encode("utf8")
        return struct.pack("<Q", len(b)) + b

    def kv(self, key, vtype, raw):
        # Callers pass the key as a plain str ("qwen3next.block_count"); kv_s() funnels
        # through here too, so accept either and always normalize to the length-prefixed form.
        if isinstance(key, str):
            key = self._str(key)
        elif isinstance(key, bytes) and not key.startswith(b"\x00"):
            # already length-prefixed by _str(): pass through untouched
            pass
        self.kv_pairs.append((key, vtype, raw))

    def kv_s(self, key, s):
        self.kv(key, KV["str"], self._str(s))

    def register(self, name, gtype, dims, size):
        self.tensors.append((name, gtype, tuple(dims), int(size)))

    # ---- layout ----
    def finalize(self):
        hdr = bytearray()
        hdr += b"GGUF"
        hdr += struct.pack("<I", 3)                     # version 3
        hdr += struct.pack("<Q", len(self.tensors))
        hdr += struct.pack("<Q", len(self.kv_pairs))
        for key, vtype, raw in self.kv_pairs:
            hdr += key + struct.pack("<I", vtype) + raw
        off = 0
        for name, gtype, dims, size in self.tensors:
            hdr += self._str(name) + struct.pack("<I", len(dims))
            for d in dims:
                hdr += struct.pack("<Q", d)
            hdr += struct.pack("<I", gtype) + struct.pack("<Q", off)
            off = _align_up(off, self.align) + size
        self._data_off = _align_up(len(hdr), self.align)
        hdr += b"\x00" * (self._data_off - len(hdr))
        self._f.write(bytes(hdr))
        self.dlen = self._data_off
        self._final = True
        return self._data_off

    # ---- data ----
    def write(self, blob):
        if not self._final:
            raise RuntimeError("write() before finalize()")
        self._f.write(blob)
        self.dlen += len(blob)

    def align_pad(self):
        pad = (-self.dlen) % self.align
        if pad:
            self.write(b"\x00" * pad)

    def close(self):
        if not self._final:
            self.finalize()
        self._f.close()


# ══════════════════ MLX affine 4-bit → GGML Q4_1 ══════════════════
#
# MLX stores affine-quantized weights as: `weight` uint32 (8 nibbles each), `scales` and
# `biases` bf16, one (scale, bias) pair per group of 64 along the last axis, with
#     dequant(q) = q * scale + bias,        q ∈ [0, 15]
#
# GGML Q4_1 stores 32 values per 20-byte block: fp16 d, fp16 m, then 16 bytes of nibbles,
#     dequant(q) = q * d + m,               q ∈ [0, 15]
#
# The two schemes are the *same affine family* — they differ only in group size (64 vs 32)
# and in how d/m are stored (fp16 vs bf16). That makes the repack a pure re-framing:
#
#   * bf16 → fp16 is EXACT for every value in fp16's range, because fp16 carries 11 mantissa
#     bits to bf16's 8. So d = (fp16)scale and m = (fp16)bias round-trip the original
#     bit patterns unchanged, and both dequantizers then evaluate the identical expression
#     `q * d + m` in fp32 — which is what makes the callers' cross-side sha gate exact
#     rather than approximate.
#   * group 64 splits cleanly into two QK41=32 blocks, each inheriting its parent's d/m.
#
# A block is only exempt from the exactness claim when fp16 cannot represent the value:
# outside fp16's range (|v| > 65504, or denormal-zero underflow), or a non-finite input.
# `pack_q41_rows` returns those as `(row, block_index)` pairs; the callers' sentinel gate
# reads `exm` to downgrade a row-0 mismatch to "EXEMPT" rather than a hard FAIL.

def mlx_q_bytes(w):
    """uint32 packed 4-bit (8 nibbles/word, low nibble = even element) → uint8 nibbles."""
    w = np.ascontiguousarray(w)
    if w.dtype != np.uint32:
        if w.dtype == np.int32:
            w = w.view(np.uint32)
        else:
            w = w.astype(np.uint32)
    n = w.shape[-1]
    out = np.empty(w.shape[:-1] + (n * 8,), np.uint8)
    wv = w.view(np.uint32)
    for i in range(8):
        out[..., i::8] = ((wv >> np.uint32(4 * i)) & np.uint32(0xF)).astype(np.uint8)
    return out


def _group_size(n, ngroups):
    if ngroups == 0:
        raise ValueError("no quantization groups")
    if n % ngroups:
        raise ValueError(f"cannot infer group size: {n} values / {ngroups} groups")
    return n // ngroups


def deq_mlx_side(q, S, B):
    """Dequantize on the MLX side: `q * scale + bias`, broadcast per group.

    `q` is (rows, K) integer nibbles; `S`/`B` are (rows, K/group) or already per-element.
    Accepts uint8 (4-bit) and int8 (8-bit) q — the 8-bit path is exact because MLX's
    affine form is identical there, only the group size differs.
    """
    q = np.asarray(q)
    Sf = np.asarray(S, dtype=np.float32)
    Bf = np.asarray(B, dtype=np.float32)
    # Expand per-group scales to per-element unless they already line up with q (or are a
    # single broadcastable scalar). The callers always hand us the per-group form.
    if Sf.shape[-1] != q.shape[-1] and Sf.shape[-1] != 1:
        g = _group_size(q.shape[-1], Sf.shape[-1])
        Sf = np.repeat(Sf, g, axis=-1)
        Bf = np.repeat(Bf, g, axis=-1)
    return q.astype(np.float32) * Sf + Bf


def deq_gguf_side(blocks):
    """Dequantize packed Q4_1 blocks back to fp32 — the GGUF side of the equality."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
    if blocks.ndim == 1:
        blocks = blocks.reshape(1, -1)
    nblocks = blocks.shape[0]
    nb = blocks.shape[1] // _Q41_BLOCK_BYTES
    # Slice per field so each view starts byte-aligned; a strided view of the interleaved
    # block is not contiguous and .view() would misread it.
    v = blocks.reshape(nblocks, nb, _Q41_BLOCK_BYTES)
    # d/m are 1 fp16 each, so after the float16 view they are already (nblocks, nb, 1) —
    # exactly the per-block broadcast shape. No extra axis.
    d = np.ascontiguousarray(v[:, :, 0:2]).view(np.float16).astype(np.float32)
    m = np.ascontiguousarray(v[:, :, 2:4]).view(np.float16).astype(np.float32)
    qs = np.ascontiguousarray(v[:, :, 4:20])                       # 16 bytes = 32 nibbles
    # ggml's layout is *split-half*, not interleaved: qs[j] low nibble = element j,
    # qs[j] high nibble = element j + qk/2. See dequantize_row_q4_1() in ggml-quants.c:
    #   y[j]        = (qs[j] & 0xF)*d + m
    #   y[j + qk/2] = (qs[j] >>   4)*d + m
    lo = (qs & np.uint8(0xF)).astype(np.uint8)                      # -> elements 0..15
    hi = ((qs >> np.uint8(4)) & np.uint8(0xF)).astype(np.uint8)      # -> elements 16..31
    q = np.empty((nblocks, nb, QK41), np.uint8)
    q[:, :, :QK41 // 2] = lo
    q[:, :, QK41 // 2:] = hi
    out = q.astype(np.float32) * d + m
    return out.reshape(nblocks, nb * QK41)


def _exact_in_fp16(v):
    """True where the fp32 value round-trips through fp16 without changing its bits.

    bf16→fp16 is exact wherever the value is representable at all (11 mantissa bits vs 8);
    the only losses are range (overflow to inf) and subnormal underflow to zero.
    """
    v = np.asarray(v, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        rt = v.astype(np.float16).astype(np.float32)
    finite = np.isfinite(v)
    return finite & (rt == v)


def pack_q41_rows(q, S, B, tag=""):
    """Pack MLX affine 4-bit rows into GGML Q4_1 blocks.

    q : (rows, K) integer nibbles (uint8) — output of `mlx_q_bytes`
    S,B : (rows, K/group) fp32 or bf16-derived scale/bias
    Returns (blocks, exempt) where blocks is uint8 (rows, nblocks*20) and exempt is a
    list of (row, block_index) whose scale/bias are not exactly representable in fp16.
    """
    q = np.asarray(q)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    rows, K = q.shape
    g = _group_size(K, S.shape[-1])
    if g % QK41:
        raise ValueError(f"MLX group size {g} is not a multiple of QK41={QK41}")
    per_group = g // QK41

    Sf = np.asarray(S, dtype=np.float32).reshape(rows, K // g)
    Bf = np.asarray(B, dtype=np.float32).reshape(rows, K // g)

    # d/m per Q4_1 block: fp16 view of the parent group's scale/bias (exact in range)
    d = np.repeat(Sf, per_group, axis=-1)
    m = np.repeat(Bf, per_group, axis=-1)
    ok = _exact_in_fp16(d) & _exact_in_fp16(m)

    nblocks = K // QK41
    qb = q.reshape(rows, nblocks, QK41).astype(np.uint8)
    # Split-half nibble order to match ggml's dequantize_row_q4_1(): elements 0..15 go in
    # the low nibbles of qs[0..15], elements 16..31 in the high nibbles.
    lo = qb[:, :, :QK41 // 2] & np.uint8(0xF)
    hi = qb[:, :, QK41 // 2:] & np.uint8(0xF)
    qs = lo | (hi << np.uint8(4))                       # 16 bytes per block

    out = np.empty((rows, nblocks, _Q41_BLOCK_BYTES), np.uint8)
    # fp16 → LE uint8 pair: view as float16 then reinterpret, keeping it C-contiguous.
    out[:, :, 0:2] = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(rows, nblocks, 2)
    out[:, :, 2:4] = np.ascontiguousarray(m.astype(np.float16)).view(np.uint8).reshape(rows, nblocks, 2)
    out[:, :, 4:20] = qs

    exempt = [(int(r), int(b)) for r, b in zip(*np.nonzero(~ok))]
    if exempt:
        print(f"[q41] {tag}: {len(exempt)}/{rows * nblocks} block(s) not exactly "
              f"fp16-representable (rounds on dequant)", file=sys.stderr)
    return out.reshape(rows, nblocks * _Q41_BLOCK_BYTES), exempt