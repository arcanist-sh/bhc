---
title: "BHC runs on WebAssembly — and we can prove how much"
status: DRAFT (for review — not published)
date: 2026-07-02
author: (you)
notes: >
  Scoped to what the differential sweep on 2026-07-02 showed (236/243 native↔WASM
  byte-identical) AFTER host-backed file IO + lines/words landed. Ready to
  publish; remaining WASM gaps (Handle API, System.Directory) are disclosed
  honestly. See the trailing EDITOR NOTE for the evidence trail.
---

# BHC runs on WebAssembly — and we can prove how much

One of the promises behind BHC is that the same Haskell should be able to run in
more places without changing what it means. Today we can show a concrete step:
**BHC compiles Haskell to WebAssembly (WASI), and the WASM output behaves the
same as the native binary across our differential test suite** — with a small,
honest set of exceptions we'll name plainly.

We're not going to tell you "WASM works" and leave it there. That's the kind of
claim this project exists to avoid. Instead, here's the measurement.

## The measurement

BHC has two backends: native (via LLVM) and WebAssembly (WASI). We run a
*differential* test harness that compiles every fixture with **both** backends,
runs each, and compares their output byte-for-byte against the expected result.
Divergence between the two backends is the signal — it catches cases where one
backend silently mishandles something the other gets right.

On 2026-07-02, across **243 fixtures** (hello-world through algebraic data types,
user-defined typeclasses with dictionary passing, monad transformers, and the
guaranteed-fusion patterns):

> **236 of 243 fixtures produce byte-identical output on native and WebAssembly.**

That covers pure computation, `Int`/`Double`/`Char`/`String` handling and
formatting, closures and thunks (laziness), constructor allocation and pattern
matching, typeclass dispatch, `putStr`/`putStrLn`/`print` to stdout, and — via
WASI — reading and writing real files on disk. Compile the same source for
`wasm32-wasi`, run it under `wasmtime`, and you get the same answer as the native
build.

## Try it

```bash
# Native
bhc Main.hs -o main && ./main

# WebAssembly (WASI)
bhc --target=wasm32-wasi Main.hs -o main.wasm && wasmtime main.wasm
```

Reading a file works the way you'd expect, with a `--dir` preopen (standard WASI
sandboxing — the guest only sees directories you hand it):

```bash
bhc --target=wasm32-wasi WordCount.hs -o wc.wasm && wasmtime --dir=. wc.wasm
```

Or run it in the browser at [bhc.raskell.io/playground](https://bhc.raskell.io/playground/) —
the playground *is* the WASM backend, compiled to run BHC itself in WASM.

## What doesn't work yet — said plainly

Differential testing is only worth doing if you report what it finds. Three honest
caveats:

**1. Some standard-library corners aren't wired on WASM yet.** File *contents*
work — `readFile`/`writeFile` go through WASI (`path_open`/`fd_read`/`fd_write`)
and `lines`/`words`/`unlines` process them — so "read a file, transform its
lines, write it back" runs. But the handle-based API (`openFile`/`hGetLine`/
`hClose`) and `System.Directory` (`doesFileExist`, …) are still unimplemented on
WASM; the native backend has them. If your program sticks to `readFile`/
`writeFile`, you're on solid ground; if it opens `Handle`s, you're not yet.

**2. The frontier is shared, not infinite.** "236/243 identical" is across *the
programs BHC already compiles* — the same language surface the native backend
handles. BHC is still a beta compiler; large real-world packages don't fully
compile on *either* backend yet. WASM is at parity with native; it is not ahead
of the compiler as a whole.

One more thing worth knowing if you're tempted to run something long-lived:
BHC's heap is not yet garbage-collected on the compiled path — short-lived
programs are fine, long-running ones will grow. That's true on both backends and
is tracked separately.

## Why this matters

The point of a WASM target isn't novelty — it's that a verification-first
compiler can now put its output somewhere a reader can check it, in a sandbox, in
a browser, with no toolchain to install. And because we test the two backends
*against each other*, "it runs on WASM" is a claim with evidence attached rather
than a vibe. That's the shape we want every BHC capability to have: not "trust
us," but "here's what we checked, and here's what we didn't."

---

<!--
EDITOR NOTE (delete before publishing):

Status: ready to publish. The file-IO gap the first draft disclosed has been
fixed (host-backed readFile/writeFile via WASI + lines/words), so this draft
claims host-backed file IO honestly. Remaining WASM gaps (Handle API,
System.Directory) are disclosed in "What doesn't work yet" and are a fair,
smaller caveat.

Evidence backing every number here (all 2026-07-02):
  - crates/bhc-e2e-tests/differential.py: 236 agree / 243 total. The harness now
    stages fixture data files into the work dir and runs `wasmtime --dir=.`.
  - Host file IO landed in bhc-wasm (wasi.rs generate_file_read_host/
    generate_file_write_host via path_open/fd_read/fd_write; lines/words in
    core_lower.rs). Verified by hand: readFile of a host input.txt prints the
    file under `wasmtime --dir=.`; writeFile persists (o.txt has content on disk);
    file_stats/file_reverse fixtures now agree with native.
  - Remaining "native right / wasm wrong": handle_io (Handle API), system_ops
    (System.Directory) — the two caveats named in the post. Other non-agrees are
    a native limitation (show_types) and stdin fixtures the harness can't feed.
  - The prior "WASM binaries fail wasmtime validation" claim (ROADMAP/CLAUDE.md)
    was stale and has been corrected in-repo.
  - GC/leak caveat: spec/BHC-REVIEW-0001 §5.1.
  - NOTE: run `cargo fmt --all -- --check` before committing the Rust changes; CI
    has failed on fmt nits before.
-->
