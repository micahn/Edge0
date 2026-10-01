// serve/mem_budget.cc — process memory budget. POSIX port of the Windows implementation.
//
// Semantics on Windows: budget = N MB ⇒ resident set is capped at N; beyond that the OS
// evicts our pages (mmap'd model-file pages first, so the next touch faults a real re-read).
//
// Linux has no equivalent primitive. RLIMIT_RSS/RLIMIT_AS are not enforced the way the
// Windows working-set quota is (RLIMIT_RSS is a no-op since Linux 2.6 on x86; RLIMIT_AS
// caps *virtual* size, which an mmapped MoE checkpoint blows past long before residency
// matters). The one mechanism that does exist is cgroup v2 memory.high / memory.max,
// which is a property of the cgroup, not the process — a process can join a cgroup
// (unshare/CLONE_NEWCGROUP + write its own cgroup.procs) if it has the privilege, but
// that is out of scope for an inference server and would silently require root.
//
// So on Linux the honest answer is: report that the budget is inert, and point at the two
// mechanisms that DO work unprivileged — the cgroup, and the edge0 pool's own arena budget
// (--pool-mb, which bounds the only allocation this codebase actually controls).
#include "mem_budget.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

E0_MB_API bool edge0_apply_mem_budget(int mb) {
    if (mb <= 0) {                                        // debug-compat env fallback
        const char * e = getenv("E0_MEM_BUDGET_MB");
        if (e && *e) mb = (int) atol(e);
    }
    if (mb <= 0) return false;

    // Report where the RSS actually stands so the number in the app's UI is not a guess.
    long page_kb = 0, rss_pages = 0;
    if (FILE * f = fopen("/proc/self/statm", "r")) {
        if (fscanf(f, "%ld %ld", &page_kb, &rss_pages) == 2) { /* got both */ }
        fclose(f);
    }
    const long rss_mb = (long) (rss_pages * (long) sysconf(_SC_PAGESIZE) >> 20);

    fprintf(stderr,
            "[mem-budget] requested %d MB, current RSS %ld MB — INERT on Linux: there is no "
            "per-process working-set cap (SetProcessWorkingSetSizeEx has no equivalent).\n"
            "             Use a cgroup v2 memory.high for a real cap, or --pool-mb to bound "
            "the edge0 expert pool arena.\n",
            mb, rss_mb);
    fflush(stderr);
    return false;
}