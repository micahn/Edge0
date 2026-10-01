# edge0-linux — Linux port of the Windows/Vulkan engine

English | [中文](README_zh.md)

This directory is a **Linux port of the `windows/` product tree**. edge0's four shipped
runtimes are macOS (Metal/MLX), iOS (Swift/MLX), Android (arm64 NEON) and Windows (C++ +
Vulkan). None of them build on Linux. This port takes the Windows path — the only one that
is not bound to Apple or ARM hardware — and replaces the Win32-specific pieces with POSIX
equivalents so the same patched llama.cpp runs on an x86_64 GPU.

**This is a community port, not an upstream-supported platform.** It is not covered by
edge0's CI, and none of the numbers in the root README apply to it. See "Caveats" below for
what does and does not carry over.

## What was actually ported

The engine itself is upstream llama.cpp — nothing edge0-specific is written in C++ here.
The port covers the three edge0-owned pieces that the `common` + `windows` patch bands
require at build time:

| edge0 piece | Linux port | Notes |
|---|---|---|
| `serve/prefetch.cc` (854 lines) | `linux/serve/prefetch.cc` | full port, not a stub |
| `serve/mem_budget.cc` | `linux/serve/mem_budget.cc` | **degraded** — see below |
| `scripts/vendor-build.ps1` | `linux/scripts/vendor-build.sh` | same depot/band contract |

The patch bands replay **unmodified**. `common` (6 patches) + `windows` (2 patches) apply
cleanly with `git am --3way` and reproduce the ledger's golden tree hash
`df9d12dcb59aa3314f3356c2cdb777ee4ca4919c` (see `patches/llama.cpp/README.md`).

### Win32 → POSIX substitutions

| Win32 | Linux | Where |
|---|---|---|
| `CreateFileA` | `open()` | expert-pool read fd |
| `CreateFileMappingA` / `MapViewOfFile` | `mmap(PROT_READ, MAP_SHARED)` | prefetch anchor view |
| `VirtualAlloc` (RESERVE, COMMIT) | `mmap(MAP_PRIVATE\|MAP_ANONYMOUS)` | pool arenas |
| `SetFilePointerEx` + `ReadFile` | `pread()` | pool fills |
| `VirtualLock` / `VirtualUnlock` | `mlock()` / `munlock()` | predicted-range pinning |
| `PrefetchVirtualMemory` | `madvise(MADV_WILLNEED)` | advisory prefetch |
| `InterlockedIncrement` / `InterlockedCompareExchange` | `__atomic_add_fetch` / `__atomic_compare_exchange_n` | pool slot state |
| `CRITICAL_SECTION` + `CONDITION_VARIABLE` | `pthread_mutex_t` + `pthread_cond_t` | slot fill/wait |
| `GetProcessWorkingSetSizeEx` / `SetProcessWorkingSetSizeEx` | *no equivalent* | memory budget |
| `QueryMemoryResourceNotification("MemoryLow")` | `MemAvailable` from `/proc/meminfo` | pool shrink |
| `_fseeki64` | `fseeko` (`off_t` is 64-bit on LP64) | GGUF header parse |
| `LONG` (32-bit `long` on Win32) | `int32_t` + `%d` formats | trace counters |
| `extern "C" static` (MSVC-legal) | plain `static` | resolver callback |

Two deviations are worth calling out because they are not mechanical:

- **`--mem-budget-mb` does nothing on Linux.** Windows caps a process's working set with
  `SetProcessWorkingSetSizeEx(QUOTA_LIMITS_HARDWS_MAX_ENABLE)`. Linux has no per-process
  equivalent: `RLIMIT_RSS` has been a no-op since 2.6, and `RLIMIT_AS` caps *virtual* size,
  which an mmapped MoE checkpoint blows past long before residency matters. The only real
  mechanism is cgroup v2 `memory.high`, which is a cgroup property and needs privilege. The
  port therefore reports the current RSS and returns `false` rather than pretending to cap.
  The alternative knob is `--pool-mb`, which does bound what the pool allocates.
- **`mlock` replaces `VirtualLock`, with the same caveat.** Both need a privilege consumer
  machines don't grant by default (`SeLockMemoryPrivilege` / `CAP_IPC_LOCK` +
  `RLIMIT_MEMLOCK`). Both are best-effort; the pool path does not depend on them.

## Prerequisites

- Linux x86_64, a Vulkan-capable GPU (tested on AMD RDNA3 / RADV; NVIDIA and Intel should work)
- CMake ≥ 3.21, `git`, a C++17 compiler (tested: GCC 16.2)
- `vulkaninfo` from `vulkan-tools`; a Vulkan ICD + driver
- Python 3.10+ with `numpy` and `pyyaml` (converter only)

## Build

```bash
# One-time: materializes vendor/llama.cpp at the pin, replays the bands, builds.
bash linux/scripts/vendor-build.sh
```

Useful flags:

```bash
bash linux/scripts/vendor-build.sh --assemble-only   # replay + hash check, no compile (~10s)
bash linux/scripts/vendor-build.sh --clean            # recreate the worktree first
bash linux/scripts/vendor-build.sh -d build-vk-j      # separate build dir (parallel builds)
```

Artifacts land in `wt/win/build-vk/bin/` — `llama-server`, `llama-cli`, `llama-bench`, and
the `lib*.so` set. The script builds with `-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release` plus
`-march=native`.

To verify the replay without paying for a build, `--assemble-only` prints the worktree tree
hash; compare it against the golden value in `patches/llama.cpp/README.md`.

## Helper scripts

| script | purpose |
|---|---|
| `linux/scripts/vendor-build.sh` | materialize llama.cpp at the pin, replay bands, build Vulkan |
| `linux/run-8b.sh` | start the 8B server with the tuned config |
| `linux/bench-server.sh` | A/B two launch configs (start, measure, tear down) |
| `scripts/fetch-tier.sh` | rate-limited resumable model fetch from Hugging Face |

## Convert a model

The checkpoints ship MLX int4 safetensors and must be converted to GGUF. The converters live
in `windows/tools/` and are pure numpy — they run fine on Linux.

```bash
python3.12 -m venv .venv && .venv/bin/pip install numpy pyyaml

# 8B tier
.venv/bin/python windows/tools/repack_r3_8b.py --dir models/edge0-8b --out models/edge0-8b-gguf

# LoRA adapter (needs the base GGUF present for its shape cross-check gate;
# the gate resolves <repo>/windows/models/<tier>-gguf/<tier>.gguf)
ln -sfn ../../models/edge0-8b-gguf windows/models/edge0-8b-gguf
.venv/bin/python windows/tools/lora_mlx_to_gguf.py --dir models/edge0-8b

# 35B tier
.venv/bin/python windows/tools/repack_r3.py --dir models/edge0-35b --out models/edge0-35b-gguf
```

`repack_r3_8b.py` exits non-zero if any gate fails (tensor-count identity, per-tensor Q4_1
sentinel sha, `ssm_ba` permutation reversibility, `exp` dual-path). Read the emitted
`<tier>-audit.json`.

> **Upstream gap:** `windows/tools/repack_mlx_to_gguf.py` — the shared conversion core that
> all four converters import — was never committed to the repository. It is reimplemented in
> this checkout (see its module docstring for the format contracts and the Q4_1 argument).
> Treat it as unverified against upstream's golden artifacts; the converters' own gates pass,
> but they cannot compare against hashes produced by the missing original.

## Run

```bash
bash linux/run-8b.sh
```

Then open **http://127.0.0.1:8081/** — llama.cpp ships its own SvelteKit web UI, served
from the server root. No separate frontend needed.

> The built-in UI is served **gzip only**. `curl http://127.0.0.1:8081/` returns
> `415 Error: gzip is not supported by this browser`; that's expected, not a failure. Use
> `curl --compressed ...` or just open a real browser.

The script's flags are the measured-best single-user chat config (see **Tuning**). Override
via `EDGE0_PORT EDGE0_CTX EDGE0_NP EDGE0_POOL_MB EDGE0_NO_LORA=1`.

For API access:

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-8b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

The model emits a long `reasoning_content` block before its answer. Skip it with
`"chat_template_kwargs": {"enable_thinking": false}` (the web UI has a toggle for this).

## Tuning

Measured on this machine (Ryzen 7 5800X, 8c/16t · RX 9070 16 GB · 31 GB RAM). Decode tok/s
via the server, LoRA adapter loaded unless noted.

| config | short prompt | 1.2k-token prefill |
|---|---:|---:|
| baseline: `-ngl 99 --pool-mb 512` | 77 | 84 |
| `+ -fa on` | 77 | 87 |
| `+ -np 1` (single slot) | 87 | **97** |
| `+ -np 1 -t 16` | 86 | 95 |
| `-np 1 -fa on`, **no LoRA** | 100 | **117** |

Reproduce any row with `bash linux/bench-server.sh "<label>" <flags…>`.

What matters, and what doesn't:

- **`-np 1` is worth ~15%.** The server defaults to 4 slots with unified KV; for a single
  user that's pure per-token overhead. This is the only change that costs nothing.
- **LoRA costs ~20% decode** (97 → 117 tok/s without it). That's per-layer adapter math on
  every token. Kept on by default because it's what recovers most of the int4 quantization
  loss; set `EDGE0_NO_LORA=1` to trade quality for speed.
- **`--pool-mb` is irrelevant here** (512 MB vs 4 GB: within noise). The pool exists to help
  when experts must fault in from SSD. With the model page-cached it's dormant weight.
- **Thread count is irrelevant** (8/12/16 all within noise) — CPU sits ~88% idle.
- **Disk I/O is not the bottleneck.** Measured 0 MB read from disk during decode; the whole
  working set is in page cache.

### Why there's no big win left

During decode this system runs at **6-9% GPU utilization with ~9 GB of VRAM free**, 88%
idle CPU, and zero disk reads. Nothing is saturated. The workload is latency-bound at the
graph-submission level rather than throughput-bound, which is why adding threads, VRAM, or
pool capacity doesn't move the number. For reference, `llama-bench` reports ~153 tok/s
`tg32` in isolation versus ~97 through the HTTP server — that gap is request-path overhead
(template rendering, sampling, SSE), not something engine flags can recover.

The two real levers are therefore: **more capable hardware** (the GPU is nearly idle, so a
faster GPU helps proportionally), and **the LoRA trade** above.

### Engine env gates

Dormant by default, same names as the Windows build:

| var | effect |
|---|---|
| `E0_POOL=sticky\|obs` | route expert reads through the L1 pool (needed with `--pool-mb`) |
| `E0_POOL_OBS_MB=<n>` | observation-zone budget for head-predicted experts |
| `E0_PREROUTER=<path>` | enable the prerouter head + advisory prefetch |
| `E0_PREFETCH=1` | issue `madvise(WILLNEED)` on predicted ranges |
| `E0_PREF_TRACE_N=<k>` | trace cadence (default 32) |
| `E0_PREF_STRIDE=<k>` | recompute the head every k steps |
| `E0_PREF_ASYNC=0` | synchronous head step instead of the worker thread |
| `E0_PIN_MAX_MB=<n>` | hot-zone byte cap (default 3072) |
| `E0_POOL_SELFCHECK=1` | byte-compare pool fills against the mmap view |
| `E0_MEM_BUDGET_MB=<n>` | inert on Linux — see above |

## Measured on this machine

AMD Radeon RX 9070 (RADV, Vulkan), 31 GB RAM, Linux 6.x, GCC 16.2. `-ngl 99 --pool-mb 512`,
`-c 8192`. **Not comparable to the root README's tables** (different hardware, different OS,
different backend); the point is that it runs and the numbers are stable.

| tier | prompt tok | prefill | decode | notes |
|---|---:|---:|---:|---|
| edge0-8b (Q4_1 + LoRA) | 409 | 1292 tok/s | 96 tok/s | `-np 4`, 4 slots |
| edge0-8b (Q4_1 + LoRA) | 2381 | 997 tok/s | 86 tok/s | `-np 4`, 4 slots |
| edge0-8b (Q4_1 + LoRA) | 1204 | 963 tok/s | 97 tok/s | `-np 1`, recommended |
| edge0-8b (Q4_1, no LoRA) | 1204 | 1287 tok/s | 117 tok/s | `-np 1`, `EDGE0_NO_LORA=1` |

Steady-state process RSS ≈ 400 MB against a 5 GB checkpoint (page cache serves the rest;
expert weights are mmapped and read on demand, which is the whole point of the design).
Peak RSS during a cold start reaches ~4.9 GB as the weights are faulted in.

Qualitatively, output is coherent: correct arithmetic (`17 × 3 → 51`), correct factual recall,
and well-formed creative output. The model also emits long `reasoning_content` blocks before
its answer — that is the trained behavior, not a defect. Pass
`"chat_template_kwargs": {"enable_thinking": false}` to skip it.

## Caveats

- **Not upstream-supported.** No CI, no Windows parity testing, no issue tracker backing.
- **`--mem-budget-mb` is inert** on Linux (no working-set cap exists). If you need a hard RSS
  bound, run the server in a cgroup with `memory.high`.
- **`mlock` pinning is best-effort** on both platforms and mostly fails unprivileged; the
  pool's real mechanism is demand fill plus the OS page cache.
- **The prerouter head weights** (`prerouter_edge0_*.safetensors`) are parsed by the ported
  `prefetch.cc`, but the head shapes are hardcoded for the 35B tier in upstream. On 8B the
  head stays dormant (`heads=0` in the INIT trace) unless the shapes are revisited — the pool
  path still works.
- **`RADV prints a conformance warning** on startup ("not a conformant Vulkan implementation,
  testing use only"). It is informational, and results above were produced with it.
- **The 35B tier has not been benchmarked here.** It converts and loads, but 31 GB of RAM
  against a ~21 GB GGUF is paging-bound; treat its numbers as unmeasured.