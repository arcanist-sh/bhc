#!/usr/bin/env python3
"""Differential backend test: run every fixture through the native and WASM
backends and compare their output.

Per-backend e2e suites check each fixture against its own `expected.txt`, but
only on the backends listed in its `test.toml`. They cannot catch cases where
the two backends *disagree* on a feature that one of them silently mishandles.
This driver compiles and runs each fixture on BOTH backends, diffs stdout, and
classifies every divergence by which backend matches `expected.txt` — turning
"I happened to notice native shows lists with spaces" into a repeatable sweep.

Usage:
    python3 crates/bhc-e2e-tests/differential.py            # summary + divergences
    python3 crates/bhc-e2e-tests/differential.py -v         # also show outputs

Requires a built `target/debug/bhc`, `wasmtime` on PATH, and (for the native
backend) LLVM_SYS_211_PREFIX set.
"""
import os, sys, subprocess, glob, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BHC = os.path.join(ROOT, "target/debug/bhc")
FIX = os.path.join(ROOT, "crates/bhc-e2e-tests/fixtures")
ENV = dict(os.environ)
ENV.setdefault("LLVM_SYS_211_PREFIX", "/opt/homebrew/opt/llvm@21")
WORK = "/tmp/bhc-differential"
VERBOSE = "-v" in sys.argv
os.makedirs(WORK, exist_ok=True)


def sources(d):
    """All .hs in fixture dir, with main.hs last (matches the e2e harness)."""
    hs = glob.glob(os.path.join(d, "*.hs"))
    hs.sort(key=lambda p: (os.path.splitext(os.path.basename(p))[0].lower() == "main", p))
    return hs


def run(cmd, stdin=None, timeout=60):
    # Run in the work dir so fixtures that writeFile to a relative path don't
    # litter the repo.
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           input=stdin, env=ENV, cwd=WORK)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -99, "", "TIMEOUT"


def backend_output(srcs, extra, runner, stdin):
    """Compile with `extra` flags, then run via `runner(artifact)`. Returns
    (ok, stdout) where ok is False on compile/run failure."""
    art = os.path.join(WORK, "out.wasm" if extra else "out")
    rc, _, _ = run([BHC, *srcs, *extra, "-o", art])
    if rc != 0:
        return False, ""
    rc, out, _ = run(runner(art), stdin=stdin)
    return rc == 0, out


def main():
    dirs = sorted({os.path.dirname(p)
                   for p in glob.glob(f"{FIX}/**/main.hs", recursive=True)})
    cats = {}
    rows = []
    for d in dirs:
        name = os.path.relpath(d, FIX)
        srcs = sources(d)
        sf = os.path.join(d, "stdin.txt")
        stdin = open(sf).read() if os.path.exists(sf) else None
        ef = os.path.join(d, "expected.txt")
        expected = open(ef).read() if os.path.exists(ef) else None

        # Stage the fixture's data files (e.g. input.txt for readFile) into the
        # work dir so BOTH backends can open them: native runs with cwd=WORK, and
        # wasm runs under `wasmtime --dir=.` (WORK). Harness metadata and sources
        # are excluded. Stale data from a prior fixture is cleared first.
        META = {"expected.txt", "stdin.txt", "test.toml"}
        for old in os.listdir(WORK):
            if old not in ("out", "out.wasm"):
                try:
                    os.remove(os.path.join(WORK, old))
                except OSError:
                    pass
        for f in os.listdir(d):
            if f in META or f.endswith(".hs"):
                continue
            src = os.path.join(d, f)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(WORK, f))

        n_ok, n_out = backend_output(srcs, [], lambda a: [a], stdin)
        w_ok, w_out = backend_output(srcs, ["--target=wasm32-wasi"],
                                     lambda a: ["wasmtime", "--dir=.", a], stdin)

        if n_ok and w_ok:
            if n_out == w_out:
                cat = "agree"
            elif expected is None:
                cat = "diff (no expected)"
            elif n_out == expected:
                cat = "diff: native right"
            elif w_out == expected:
                cat = "diff: wasm right"
            else:
                cat = "diff: both wrong"
        elif n_ok and not w_ok:
            cat = "wasm fails"
        elif w_ok and not n_ok:
            cat = "native fails"
        else:
            cat = "both fail"

        cats[cat] = cats.get(cat, 0) + 1
        if cat != "agree":
            rows.append((cat, name, n_out, w_out, expected))

    print("=== differential backend summary ===")
    for k in sorted(cats):
        print(f"  {k}: {cats[k]}")
    print(f"  total: {sum(cats.values())}")
    print("\n=== divergences ===")
    for cat, name, n, w, e in sorted(rows):
        print(f"  [{cat}] {name}")
        if VERBOSE:
            print(f"      native:   {n!r}")
            print(f"      wasm:     {w!r}")
            print(f"      expected: {e!r}")


if __name__ == "__main__":
    main()
