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
   parsec fails parsec. `deps` therefore compiles the override modules LAST.
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
4. **A probe that hangs for ~40s before running is macOS Gatekeeper**, not a
   compiler bug; static linking (no custom dylibs) avoids the wedged
   `syspolicyd` path entirely.

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
