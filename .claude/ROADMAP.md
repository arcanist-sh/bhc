# BHC Roadmap

**Document ID:** BHC-ROAD-0002
**Status:** Active
**Last Updated:** 2026-02-27

---

## Where We Are

BHC parses 100% of Pandoc's 221 source files. The parser, layout rule, type
checker, Core IR optimizer, and LLVM codegen pipeline are battle-tested across
199 E2E tests and 70 implementation milestones. Separate compilation (`.bhi`
interfaces, `-c` mode, `--package-db`) is wired. The hx package manager
integration compiles.

What remains between here and compiling real Hackage packages — and ultimately
Pandoc — is connecting modules together: import resolution across packages,
compiling dependency trees, and filling library API gaps.

This roadmap replaces the previous M0–M11 milestone structure with a
forward-looking plan organized around concrete deliverables.

---

## Milestone Summary

| Milestone | Name | Status | Target |
|-----------|------|--------|--------|
| P1 | Multi-Module Pandoc Check | 🔴 Not started | Next |
| P2 | Hackage Package Compilation | 🔴 Not started | — |
| P3 | Pandoc `bhc check` | 🔴 Not started | — |
| P4 | Pandoc Native Binary | 🔴 Not started | — |
| W1 | WASM Backend Validation | 🟡 Blocked | — |
| T1 | Tooling Polish | 🟡 Partial | — |

---

## P1 — Multi-Module Pandoc Check

**Goal:** `bhc check` succeeds on 50+ Pandoc modules by resolving cross-module
imports within Pandoc's own source tree.

**Why this is next:** Today 10/221 Pandoc modules pass `bhc check`. The other
211 fail exclusively on unresolved imports — not parse errors, not type errors.
This milestone connects Pandoc's modules to each other.

### Deliverables

- [ ] **Import resolution from source tree** — When `Text.Pandoc.Slides`
  imports `Text.Pandoc.Definition`, BHC finds and loads the module from the
  same source directory. This requires wiring the `-I` flag or an equivalent
  source search path into the `bhc check` pipeline.

- [ ] **Dependency-ordered batch check** — `bhc check` on a directory compiles
  modules in topological order (already implemented for multi-file compilation;
  extend to `check` mode).

- [ ] **Cross-module type propagation in check mode** — Types, constructors,
  and class instances from compiled modules must be visible when checking
  downstream modules. The `.bhi` pipeline handles this for separate compilation;
  verify it works in batch-check mode.

- [ ] **Re-export resolution** — `module Text.Pandoc.Class` re-exports names
  from `Text.Pandoc.Class.PandocMonad`. BHC must follow `module Foo` re-exports
  in export lists.

### Exit Criteria

- `bhc check` on the Pandoc source tree (221 files) passes for all modules
  whose imports are satisfiable within the Pandoc source tree itself (estimated
  50–80 modules, depending on how many reference only internal modules)
- Zero regressions in existing 199 E2E tests

### Key Files

```
crates/bhc-driver/src/lib.rs      — batch check mode, source search paths
crates/bhc-lower/src/loader.rs    — module loading and re-export following
crates/bhc-interface/src/         — .bhi generation/consumption
```

---

## P2 — Hackage Package Compilation

**Goal:** Compile Pandoc's leaf dependencies from Hackage source using `hx build`.

**Why:** Pandoc imports ~80 packages. We can't stub them all. The fastest path
is compiling the simple leaf packages (no C FFI, no TH) from source.

### Deliverables

- [ ] **End-to-end `hx build` test** — Pick a trivial Hackage package (e.g.,
  `data-default-class`, `tagged`, `void`) and compile it from source using the
  full hx pipeline: fetch → solve → build with BHC.

- [ ] **`base` package stub** — BHC's 500+ builtins must be exposed as a
  proper `base` package in the package DB so that Hackage packages can
  `import Data.List`, `import Control.Monad`, etc. and resolve against BHC's
  built-in implementations.

- [ ] **`text` / `bytestring` / `containers` package stubs** — Same treatment.
  BHC already has RTS implementations; they need package DB entries with correct
  module maps so downstream packages can import them.

- [ ] **Conditional dependencies** — Many `.cabal` files use `if impl(ghc)`
  guards. hx-solver needs to handle BHC as a known implementation, or fall
  through to a default branch.

- [ ] **Compile 10 leaf packages** — Target list (all zero-TH, minimal deps):
  `data-default-class`, `tagged`, `void`, `nats`, `semigroups`,
  `hashable`, `unordered-containers`, `scientific`, `attoparsec`,
  `case-insensitive`.

### Exit Criteria

- `hx build` compiles at least 10 Hackage packages from source to `.bhi` + `.o`
- Packages are usable as dependencies by downstream packages
- No manual patching of package source required (or patches are minimal and
  documented)

### Key Files

```
crates/bhc-package/src/           — package DB, package resolution
crates/bhc-interface/src/         — .bhi generation for compiled packages
hx repo: hx-bhc/                  — BHC backend for hx package manager
```

---

## P3 — Pandoc `bhc check`

**Goal:** `bhc check` succeeds on 100% of Pandoc's 221 source files.

**Why:** This is the exit criterion from the blog post promise. It proves BHC
can type-check a real 60k LOC Haskell project.

### Deliverables

- [ ] **pandoc-types package** — Compile or stub `Text.Pandoc.Definition`
  (defines `Block`, `Inline`, `Meta`, etc.). This is Pandoc's core type
  package and is imported by nearly every module.

- [ ] **parsec compatibility** — Pandoc uses parsec combinators extensively.
  Either compile parsec from source (preferred) or provide a compatible API
  stub. Key types: `ParsecT`, `SourcePos`. Key combinators: `parse`, `try`,
  `many`, `many1`, `char`, `string`, `noneOf`, `oneOf`, `<|>`, `option`,
  `optional`, `sepBy`, `endBy`, `between`, `choice`.

- [ ] **aeson compatibility** — Pandoc uses aeson for metadata. With
  GHC.Generics working, generic deriving should cover `ToJSON`/`FromJSON`.
  Compile aeson from source or provide the core `Value` type + `encode`/`decode`.

- [ ] **Remaining Pandoc deps** — Compile or stub: `skylighting-core`,
  `doctemplates`, `texmath`, `xml-conduit`, `network-uri`, `http-types`,
  `zip-archive`, `citeproc`, `commonmark`.

- [ ] **RankNTypes / ExistentialQuantification** — Some Pandoc modules use
  rank-2 types. Implement at least rank-2 (forall in argument position) to
  unblock these.

### Exit Criteria

- `bhc check` on all 221 Pandoc 3.6.4 source files: 221/221 pass
- Automated CI job runs this check on every commit

### Key Files

```
crates/bhc-typeck/src/            — RankNTypes support
crates/bhc-parser/src/            — any remaining parse gaps
```

---

## P4 — Pandoc Native Binary

**Goal:** `bhc build pandoc` produces a working native executable.

**Why:** This is the ultimate proof that BHC compiles real-world Haskell.

### Deliverables

- [ ] **Full codegen for Pandoc modules** — Move from `check` to `build`.
  Every Core IR construct used by Pandoc must lower to LLVM.

- [ ] **Link 221 modules** — Produce a single native binary from all compiled
  object files plus BHC's RTS.

- [ ] **Functional smoke test** — `echo "# Hello" | ./pandoc -f markdown -t html`
  produces `<h1>Hello</h1>`.

- [ ] **Correctness test suite** — Run Pandoc's own test suite against the
  BHC-compiled binary. Track pass rate.

### Exit Criteria

- `bhc build` on Pandoc source tree produces a native binary
- Binary converts Markdown to HTML correctly for basic documents
- Pandoc's test suite pass rate documented (target: >80%)

---

## W1 — WASM Backend Validation

**Goal:** `bhc --target=wasi Main.hs -o app.wasm && wasmtime app.wasm` works
for hello world.

**Status:** WASM emitter exists (bhc-wasm) but output fails wasmtime validation.
All 6 WASM E2E tests fail with "WebAssembly translation error".

### Deliverables

- [ ] **Fix binary format** — Debug wasmtime validation errors. The emitter
  produces output but the WASM module structure is malformed.
- [ ] **Hello world E2E** — `putStrLn "Hello"` compiles to WASM and runs under
  wasmtime.
- [ ] **Numeric kernels in WASM** — Verify Tensor IR → Loop IR → WASM pipeline
  produces correct results.

### Exit Criteria

- `wasmtime app.wasm` prints "Hello, World!" for a BHC-compiled WASM binary
- At least 3 of 6 WASM E2E tests pass

---

## T1 — Tooling Polish

**Goal:** REPL, IR inspector, and LSP provide usable developer experience.

### Deliverables

- [ ] **REPL evaluation** — `bhci` evaluates expressions (currently stubbed).
  At minimum: literals, function application, let-bindings, `:type` command.
- [ ] **IR inspector** — `bhi` displays Core IR and Tensor IR for compiled
  modules. Useful for debugging optimization passes.
- [ ] **LSP basics** — Go-to-definition, hover-for-type, diagnostics on save.
  The LSP server compiles but needs integration testing.
- [ ] **`-ddump-*` flags** — Wire `dump-core-after-simpl`, `dump-core-after-demand`,
  `dump-core-final` into the driver for optimization debugging.

### Exit Criteria

- `bhci` evaluates `1 + 1` and prints `2`
- `bhi` displays Core IR for a compiled module
- LSP provides hover types in VS Code

---

## Remaining Work (Carried Forward)

Items from the previous roadmap that are not yet complete but don't warrant
their own milestone. These will be addressed opportunistically or as blockers
surface.

### Language Features

| Feature | Status | Notes |
|---------|--------|-------|
| RankNTypes (rank-2) | 🔴 | Needed for some Pandoc modules |
| ExistentialQuantification | 🔴 | Used in Pandoc's PandocMonad |
| ConstraintKinds | 🔴 | Used in some Pandoc deps |
| TemplateHaskell | 🔴 | Deferred — GHC.Generics covers most TH use cases |
| `.hs-boot` files | 🔴 | Mutual module recursion |
| `foreign export` codegen | 🔴 | `foreign import` works; export not yet |
| Typed holes (`_ :: Type`) | 🔴 | Nice for IDE experience |
| Error recovery in layout | 🔴 | Better error messages |
| `INLINE`/`SPECIALIZE` pragmas | 🟡 | Parsed but not used by optimizer |
| Incremental recompilation | 🔴 | Check timestamps/hashes |

### Standard Library Gaps

| Library | Status | Notes |
|---------|--------|-------|
| Data.Graph / Data.Tree | 🔴 | Used by some Pandoc deps |
| process (spawn subprocesses) | 🔴 | `System.Process` |
| time (date/time types) | 🔴 | `Data.Time` |
| network-uri (URI parsing) | 🔴 | Small, pure Haskell |
| Data.Char full Unicode | 🟡 | Currently ASCII-only predicates |
| `realToFrac` | 🔴 | Numeric conversion |
| `reads` (general parsing) | 🔴 | Used by Read instances |
| Temporary files | 🔴 | `withTempFile`, `withTempDirectory` |

### Backend/Runtime

| Item | Status | Notes |
|------|--------|-------|
| WASM binary validation | 🔴 | See W1 |
| GPU end-to-end testing | 🟡 | Requires CUDA hardware |
| Bare metal codegen | 🟡 | LLVM target work needed |
| REPL evaluation | 🔴 | See T1 |

---

## Completed Work

Everything below is done. Kept for reference.

### Compiler Pipeline (M0–M3)

- Lexer, parser, type checker, HIR, Core IR, LLVM codegen — all working
- 199 E2E tests across 70 milestones (E.1–E.70)
- Native executables via LLVM (hello world through GADTs + monad transformers)
- Closures, thunks, lazy evaluation, pattern matching, ADTs
- Generational GC, work-stealing scheduler, STM

### Language Features (M11 / Phase 9)

- 30+ GHC extensions implemented
- Full typeclass system: dictionary passing, superclasses, default methods,
  DeriveAnyClass, GND, DerivingStrategies
- 9 stock derivable classes: Eq, Show, Ord, Enum, Bounded, Functor, Foldable,
  Traversable, Read + Generic stubs
- Record syntax: named fields, accessors, construction, update, wildcards, puns
- GADTs with type refinement
- Type families (open, closed, associated) + data families
- CPP preprocessing (built-in Rust preprocessor)
- Layout rule (full Haskell 2010 Section 10.3)
- Type applications (`f @Int x`)
- Pattern synonyms, view patterns, multi-way if, lambda-case
- Standalone deriving, empty data decls, strict data, default signatures

### Core IR Optimizer (O.1–O.4)

- Simplifier: constant folding, beta reduction, case-of-known-constructor,
  case-of-case, dead binding elimination, inlining, occurrence analysis
- Pattern match compilation: Augustsson/Sestoft decision trees, exhaustiveness
  and overlap checking
- Demand analysis + worker/wrapper: boolean-tree strictness, fixpoint iteration
- Dictionary specialization: direct method selection on known dictionaries

### Standard Library

- 500+ builtins: Prelude, Data.List (70+), Data.Map/Set/IntMap/IntSet,
  Data.Maybe, Data.Either, Control.Monad, Data.Char, Data.Text (25+),
  Data.ByteString (24), monad transformers (StateT/ReaderT/ExceptT/WriterT),
  Data.Sequence, GHC.Generics, IORef, Rational
- Lazy Text, Lazy ByteString, ByteString.Builder
- File IO, directory operations, exception handling with typed catch
- Text.Encoding: encodeUtf8/decodeUtf8

### Separate Compilation (E.66)

- `-c` mode: compile to `.o` + `.bhi` without linking
- `--odir`/`--hidir`/`--package-db` flags
- `.bhi` interface generation and consumption
- TypeConverter bridge for cross-module type checking
- hx package manager integration wired (hx-bhc crate)

### Runtime System (M5, M7)

- Structured concurrency: withScope, spawn, await, cancel, deadlines
- STM: TVar, atomically, retry, orElse, TMVar, TQueue
- Work-stealing scheduler with crossbeam deques
- Incremental GC with tri-color marking, SATB barriers
- Realtime profile: bounded GC pauses, frame arenas
- Embedded profile: no-GC static allocation

### Numeric Pipeline (M1–M3)

- Tensor IR with shape/stride metadata and fusion passes
- Loop IR with auto-vectorization (SIMD) and parallel loops
- Hot arena allocator, pinned buffers
- GPU backend: PTX codegen, device memory, kernel launch (mock-tested)

### Pandoc Smoke Test (2026-02-27)

- 221/221 Pandoc source files parse successfully (0 parse errors)
- 10 modules pass full `bhc check` (parse + typecheck + Core IR lowering)
- 211 modules fail only on unresolved external package imports
- Parser bug fixed: `where` on its own line after module export list

---

## Principles

1. **Real code drives the roadmap** — Every milestone is validated against
   Pandoc or Hackage packages, not synthetic tests
2. **Bottom-up** — Compile leaf packages first, work up the dependency tree
3. **No stubs where compilation works** — Prefer compiling from source over
   hand-written stubs
4. **One blocker at a time** — Fix the highest-impact blocker, re-test, repeat

---

## References

- [Pandoc Smoke Test Results](TODO-pandoc.md) — Detailed error catalog
- [BHC Specification](../CLAUDE.md) — Project overview and architecture
- [Coding Rules](rules/) — Style, testing, optimization, IR design
- [Blog: BHC Parses All of Pandoc](https://arcanist.sh/bhc/blog/bhc-parses-all-of-pandoc/)
