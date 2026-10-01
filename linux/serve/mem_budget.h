// serve/mem_budget.h — process memory budget (first-class product feature).
// Independent of the prefetch router: it only caps physical residency; no prediction.
#pragma once

// POSIX build (no __declspec); ggml/llama are static libs here, so plain visibility suffices.
#define E0_MB_API __attribute__((visibility("default")))

// Primary channel: CLI flag (--mem-budget-mb → mb > 0); when mb <= 0, falls back to
// the E0_MEM_BUDGET_MB env var (debug compat). mb > 0 asks for this process's resident
// set to be held at that size. Returns whether the cap was applied.
//
// Linux caveat: see mem_budget.cc — there is no per-process working-set hard cap in
// Linux (no SetProcessWorkingSetSizeEx equivalent), so this is advisory unless run under
// a memory cgroup. It returns false and says so rather than silently doing nothing.
E0_MB_API bool edge0_apply_mem_budget(int mb);