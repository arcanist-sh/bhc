#!/usr/bin/env python3
"""Differential test against GHC: run every fixture under BHC and under GHC and
compare their output.

The existing suites check each fixture against its own `expected.txt`, and
`differential.py` checks the native and WASM backends against each other. Both
share a blind spot: `expected.txt` is only as good as whoever wrote it, and two
BHC backends can agree with each other and still both be wrong. Neither can tell
you that BHC *means* something different from Haskell.

GHC can. This driver compiles and runs each fixture with both compilers and
treats GHC's output as ground truth, so a divergence is a semantic bug in BHC
rather than a difference of opinion. It also reports where GHC disagrees with a
fixture's own `expected.txt`, which means the expectation is wrong and every
suite asserting it has been asserting a falsehood.

Why this exists: on 2026-09-02 a change to guard compilation broke
`tier3_io/pattern_guards` — a fixture that COMPILES and then crashes. The pandoc
harness (which only compiles), the ladder and the battery all stayed green for
hours. Compiling is not running, and running against a self-authored expectation
is not the same as running against Haskell.

Usage:
    python3 crates/bhc-e2e-tests/ghc_differential.py            # summary + divergences
    python3 crates/bhc-e2e-tests/ghc_differential.py -v         # also show outputs
    python3 crates/bhc-e2e-tests/ghc_differential.py DIR        # sweep DIR instead
                                                                # (e.g. ad-hoc probes)

Exit status is non-zero when BHC disagrees with GHC, or fails on something GHC
builds, so this is usable as a gate — it runs as the `differential` CI job.

`KNOWN_FAILURES` lists the fixtures BHC is known to get wrong, each with its
reason. Those are reported but do not fail the build, so the debt stays visible
without blocking. A known failure that starts PASSING also fails the run, so the
list is pruned rather than left to rot.

Requires a built `target/debug/bhc` and a GHC. The GHC is found via $BHC_GHC,
then hx's managed toolchain, then PATH.
"""
import os, sys, subprocess, glob, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BHC = os.path.join(ROOT, "target/debug/bhc")
ENV = dict(os.environ)
ENV.setdefault("LLVM_SYS_211_PREFIX", "/opt/homebrew/opt/llvm@21")

# A BHC-produced binary links against the stdlib shared objects in target/, and
# it is run from a scratch directory, so the loader needs to be told where they
# are. `bhc-e2e-tests/src/native.rs` does the same thing for the Rust harness.
# Without it every fixture touching Data.Map, Data.Text or Data.Char dies with
# "libbhc_containers.so: cannot open shared object file" — on Linux only, which
# is how it reached CI green from a macOS machine.
_LIB_VAR = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
_LIB_DIRS = [os.path.join(ROOT, d) for d in ("target/debug", "target/release")]
_LIB_DIRS = [d for d in _LIB_DIRS if os.path.isdir(d)]
if _LIB_DIRS:
    ENV[_LIB_VAR] = os.pathsep.join(_LIB_DIRS + ([ENV[_LIB_VAR]] if ENV.get(_LIB_VAR) else []))
VERBOSE = "-v" in sys.argv

BHC_WORK = "/tmp/bhc-ghcdiff/bhc"
GHC_WORK = "/tmp/bhc-ghcdiff/ghc"
META = {"expected.txt", "stdin.txt", "test.toml"}

# Fixtures BHC is known to get wrong, with the reason. Listed here so the run
# can gate CI without the debt going silent: a known failure is reported and
# does not fail the build, and a known failure that starts PASSING is reported
# too, so this list gets pruned instead of rotting.
#
# Nothing belongs here that has not been diagnosed. "It fails and I do not know
# why" is a red build, not an entry.
KNOWN_FAILURES = {
    "tier3_io/stdin_echo":
        "native stdin read path segfaults; pre-existing, see the #[ignore] on "
        "test_tier3_milestone_d_csv_parser_native",
    "tier3_io/stdin_readln":
        "same native stdin read path",
}


def find_ghc():
    """A GHC to compare against: $BHC_GHC, else hx's managed toolchain, else PATH.

    hx keeps its toolchains under ~/.hx/toolchains/ghc/<version>/bin, which is
    where a machine that builds with hx already has one — no separate install.
    """
    if ENV.get("BHC_GHC"):
        return ENV["BHC_GHC"]
    managed = sorted(glob.glob(os.path.expanduser("~/.hx/toolchains/ghc/*/bin/ghc-*")))
    managed = [p for p in managed if os.access(p, os.X_OK) and "pkg" not in p and "ghci" not in p]
    if managed:
        return managed[-1]
    return shutil.which("ghc") or ""


GHC = find_ghc()


def sources(d):
    """All .hs in a fixture dir, main.hs last — the order the e2e harness uses."""
    hs = glob.glob(os.path.join(d, "*.hs"))
    hs.sort(key=lambda p: (os.path.splitext(os.path.basename(p))[0].lower() == "main", p))
    return hs


def run(cmd, cwd, stdin=None, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           input=stdin, env=ENV, cwd=cwd)
        return r.returncode, r.stdout, r.stderr
    except (subprocess.TimeoutExpired, OSError):
        return -99, "", "TIMEOUT"


def stage(d, work):
    """Put the fixture's data files where a run can open them.

    A fixture may `readFile "input.txt"`, so its non-source, non-metadata files
    have to sit in the working directory. Stale files from the previous fixture
    are cleared first, or one fixture's data silently satisfies the next one.
    """
    os.makedirs(work, exist_ok=True)
    for old in os.listdir(work):
        p = os.path.join(work, old)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass
    for f in os.listdir(d):
        src = os.path.join(d, f)
        if f in META or f.endswith(".hs") or not os.path.isfile(src):
            continue
        shutil.copy(src, os.path.join(work, f))


def bhc_output(srcs, stdin):
    art = os.path.join(BHC_WORK, "out")
    rc, _, err = run([BHC, *srcs, "-o", art], BHC_WORK)
    if rc != 0:
        return False, "", f"compile: {err.strip()[:200]}"
    rc, out, err = run([art], BHC_WORK, stdin=stdin)
    if rc != 0:
        return False, out, f"run: rc={rc} {err.strip()[:200]}"
    return True, out, ""


def ghc_output(srcs, stdin):
    """Compile with GHC in its own directory.

    The sources are COPIED rather than compiled in place: GHC writes .hi and .o
    beside each source, and dropping build artefacts into the fixture tree would
    make the repo dirty and confuse the next run.
    """
    for s in srcs:
        shutil.copy(s, os.path.join(GHC_WORK, os.path.basename(s)))
    names = [os.path.basename(s) for s in srcs]
    art = "ghc_out"
    rc, _, err = run([GHC, "-v0", "-o", art, *names], GHC_WORK)
    if rc != 0:
        return False, "", f"compile: {err.strip()[:200]}"
    rc, out, err = run([os.path.join(GHC_WORK, art)], GHC_WORK, stdin=stdin)
    if rc != 0:
        return False, out, f"run: rc={rc} {err.strip()[:200]}"
    return True, out, ""


def main():
    if not GHC:
        print("no GHC found — set $BHC_GHC, or install one via hx", file=sys.stderr)
        return 2
    if not os.path.exists(BHC):
        print(f"no bhc at {BHC} — cargo build first", file=sys.stderr)
        return 2

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    fix = os.path.abspath(args[0]) if args else os.path.join(ROOT, "crates/bhc-e2e-tests/fixtures")

    os.makedirs(BHC_WORK, exist_ok=True)
    os.makedirs(GHC_WORK, exist_ok=True)
    print(f"bhc: {BHC}")
    print(f"ghc: {GHC}")
    print(f"fixtures: {fix}\n")

    dirs = sorted({os.path.dirname(p)
                   for p in glob.glob(f"{fix}/**/main.hs", recursive=True)})
    if not dirs:
        # A flat directory of standalone probes rather than fixture dirs.
        dirs = sorted({os.path.dirname(p) for p in glob.glob(f"{fix}/*.hs")})

    cats, rows, bad_expectations = {}, [], []
    known_rows, fixed_rows = [], []
    for d in dirs:
        name = os.path.relpath(d, fix) or os.path.basename(d)
        srcs = sources(d)
        if not srcs:
            continue
        sf = os.path.join(d, "stdin.txt")
        stdin = open(sf).read() if os.path.exists(sf) else None
        ef = os.path.join(d, "expected.txt")
        expected = open(ef).read() if os.path.exists(ef) else None

        stage(d, BHC_WORK)
        stage(d, GHC_WORK)
        b_ok, b_out, b_err = bhc_output(srcs, stdin)
        g_ok, g_out, g_err = ghc_output(srcs, stdin)

        if not g_ok:
            # GHC cannot build it, so there is no ground truth to compare
            # against. Usually a fixture that leans on a BHC-only extension or
            # profile. Reported, never counted as a BHC failure.
            cat = "ghc cannot build (skipped)"
        elif b_ok and b_out == g_out:
            cat = "agree"
        elif b_ok:
            cat = "DIVERGE"
        else:
            cat = "bhc fails"

        # An expectation GHC contradicts is a lie every other suite asserts.
        if g_ok and expected is not None and g_out != expected:
            bad_expectations.append((name, g_out, expected))

        known = name in KNOWN_FAILURES
        if known and cat in ("DIVERGE", "bhc fails"):
            cat = "known failure"
        elif known:
            cat = "KNOWN FAILURE NOW PASSES"

        cats[cat] = cats.get(cat, 0) + 1
        if cat in ("DIVERGE", "bhc fails"):
            rows.append((cat, name, b_out, g_out, b_err))
        elif cat == "known failure":
            known_rows.append((name, KNOWN_FAILURES[name]))
        elif cat == "KNOWN FAILURE NOW PASSES":
            fixed_rows.append(name)

    print("=== bhc vs ghc ===")
    for k in sorted(cats):
        print(f"  {k}: {cats[k]}")
    print(f"  total: {sum(cats.values())}")

    if rows:
        print("\n=== divergences (ghc is ground truth) ===")
        for cat, name, b, g, err in sorted(rows):
            print(f"  [{cat}] {name}")
            if cat == "bhc fails":
                print(f"      {err}")
            if VERBOSE or cat == "DIVERGE":
                print(f"      bhc: {b!r}")
                print(f"      ghc: {g!r}")

    if bad_expectations:
        print("\n=== expected.txt contradicted by GHC ===")
        for name, g, e in sorted(bad_expectations):
            print(f"  {name}")
            print(f"      ghc:      {g!r}")
            print(f"      expected: {e!r}")

    if known_rows:
        print("\n=== known failures (not gating) ===")
        for name, why in sorted(known_rows):
            print(f"  {name}: {why}")

    if fixed_rows:
        print("\n=== known failures that now PASS — remove from KNOWN_FAILURES ===")
        for name in sorted(fixed_rows):
            print(f"  {name}")

    failures = cats.get("DIVERGE", 0) + cats.get("bhc fails", 0)
    # A stale KNOWN_FAILURES entry is also a failure: it means the list is
    # claiming something is broken that is not, and the next real regression
    # there would be swallowed.
    return 1 if failures or fixed_rows else 0


if __name__ == "__main__":
    sys.exit(main())
