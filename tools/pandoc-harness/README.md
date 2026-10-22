# Pandoc integration harness

The rig that measures bhc against pandoc: build a package DB out of pandoc's
221 library modules, then compile/link/run small probe programs against it.

It used to live outside the repository (`~/Development/pandoc-harness`) and was
deleted on 2026-10-08 together with the pandoc sources, taking a week of
vendored stand-in modules with it. The scripts live here now so only the
third-party *sources* have to be fetched again.

## What you need next to it

| Path (default) | What it is |
| --- | --- |
| `$PANDOC` = `~/Development/pandoc-3.6.4/src` | pandoc 3.6.4 library sources |
| `$PTYPES` = `~/Development/pandoc-types-1.23.1.2/src` | pandoc-types sources |
| `$PARSEC` = `$HARNESS/parsec-3.1.17.0/src` | parsec sources, copied to `parsec-vendored` with `OPTIONS_HADDOCK` stripped from `Prim.hs` (open bhc header bug) |
| `$VENDOR` = `~/Development/pandoc-vendor` | hand-written stand-ins for externals bhc stubs: `System/Console/GetOpt.hs`, `Data/Version.hs`, `Paths_pandoc.hs`, `Text/Collate/Lang.hs`, `Text/Pandoc/Definition.hs` (SimpleFigure-free) |
| `$HARNESS/blaze-vendored` | clean-room `Text.Blaze.*` + a minimal `Data.Semigroup` (`Min`/`Max`) |

`bhc` and the four static libs are snapshotted into `snap/` so a probe is never
linked against a half-written `target/debug`.

## Hard-won rules

1. **A module in the package DB overrides bhc's stub list for its WHOLE export
   surface.** That is the escape hatch for a stubbed external — and the trap:
   the vendored `Data.Semigroup` exports `Min`/`Max` but not `Semigroup(..)`'s
   `sconcat`/`stimes`, which parsec's `Prim` imports, so compiling it before
   parsec fails parsec. `deps` therefore compiles the override modules LAST —
   and, because a RE-build starts with them already in the DB, deletes their
   objects first (otherwise `deps` only works on an empty DB, and a rebuild
   silently leaves `Prim`/`Token` at the objects of the previous compiler).
   Same shape of bug: vendored `Paths_pandoc` must export `getDataFileName`
   (Class.IO reads `abbreviations`, templates and translations through it), or
   Class.IO, PandocIO, PDF and the three App modules all fail to lower.
2. **Localize `_show_desc_*` once per DB build, not per probe link.** codegen
   emits that symbol per module with external linkage, so a link needs it
   localized — but doing it per probe is nearly all of a 4-minute link.
   `flink.sh localize` does the DB once and each probe link only its own
   objects: 3 seconds per probe. `llvm-objcopy` takes ONE input file, so
   `xargs -n 16` silently does nothing (and `-n 2` would overwrite inputs).
3. **Gate on values, not counts.** `ladder.sh`/`battery.sh` compare each probe's
   output against its expected string, because this campaign has twice been
   fooled by output that looked green (`OK ` passing for `OK a`).
4. **Never run cargo and a sweep at the same time.** This machine has 16 GB:
   `cargo build` with its ~17 parallel rustc processes plus a sweep's `bhc`
   compiles gets both SIGKILLed, and a killed sweep looks exactly like a
   compiler regression (`Killed: 9` against every module).
5. **`pgrep -f 'chain.sh sweep'` matches the waiting shell itself.** An
   `until ! pgrep -qf …` loop around the sweep never exits; use a bracket
   (`chain[.]sh`) or wait on the task instead.
6. **A probe that hangs for ~40s before running is macOS Gatekeeper**, not a
   compiler bug; static linking (no custom dylibs) avoids the wedged
   `syspolicyd` path entirely.

## The scripts

| Script | What it does |
| --- | --- |
| `chain.sh` | builds the package DB: `snapshot`, `deps`, `sweep` |
| `flink.sh` | compiles + links + runs one probe against the DB (~3 s) |
| `ladder.sh` | runs the value-checked probe ladder |
| `mdbis.sh` | patches pandoc's `parseMarkdown` body, recompiles that one module and relinks `QR4` (~15 s) — the bisection tool |
| `mdladder.sh` | walks `mdbis.sh` up the reader, one rung per level (`symbol`, `inline`, `many1 inline`, `inlines1`, `plain`, `block`, `parseBlocks`, full) |
| `resweep.sh` | re-runs only the modules that failed |
| `sexp_core.py` | turns a `BHC_DUMP_BIND` Rust-Debug dump into one-line sexps |
| `topo.py` | pandoc's modules in dependency order |

Useful switches while bisecting: `BHC_DUMP_BIND=<name>` (Core for one bind),
`BHC_DBG_DISPATCH=1` (which instance each class method resolved to, and why a
monadic rewrite declined), `BHC_DBG_NULLDICT=1` (every dictionary that resolved
to NOTHING — its placeholder is a zero literal, which reaches the runtime as
the tagged small int `0x1`, the signature of most crashes here), and
`RUST_BACKTRACE=1` on the probe binary, which works whenever the crash lands in
an RTS function rather than in generated code.

## Usage

```sh
./chain.sh snapshot          # copy bhc + static libs out of target/debug
./chain.sh deps              # parsec, pandoc-types, then the override modules
./chain.sh sweep             # the 221-module sweep, to convergence
./resweep.sh                 # recompile only what still fails (after a fix)
./flink.sh localize          # once per DB build
./flink.sh Probe.hs [args]   # compile, link and run one probe (~3s)
LINKER=$PWD/flink.sh ./ladder.sh   # the parser ladder, values checked
```

Set `SNAP`/`DB`/`LOGS`/`FAILS` to keep several builds side by side — a flag-on
and a flag-off sweep must never share a fail list.
