# Basel Haskell Compiler (BHC)

**Codename:** BHC
**Document ID:** BHC-SPEC-0001
**Scope:** Reference compiler + runtime for the Haskell 2026 Platform
**Primary Mission:** Modern runtime contract + world-class numeric performance

---

## Project Identity

BHC ("Basel Haskell Compiler") is a clean-slate Haskell compiler and runtime, built to remain compatible with the spirit of Haskell while introducing modern profiles (Default/Server/Numeric/Edge/Realtime/Embedded), a standardized runtime contract, and a tensor-native compilation pipeline.

BHC is named after Basel, Switzerland — as a deliberate successor culture to the Glasgow Haskell Compiler lineage, with a new focus on predictability, concurrency, and numerical computing.

### Philosophy

BHC prioritizes **predictability over folklore**: if performance matters, the compiler tells you what happened. If concurrency matters, cancellation is structured. If numerics matter, fusion is guaranteed and kernels are traceable.

### One-Liner

> BHC makes Haskell a serious 2026 systems and numeric platform — without sacrificing purity.

---

## Quick Start

```bash
# Install BHC
curl -fsSL https://bhc.raskell.io/install.sh | sh

# Compile in Haskell 2010 mode
bhc --edition=Haskell2010 Main.hs

# Or use GHC2024 compatibility
bhc --edition=GHC2024 Main.hs

# Use numeric profile for performance-critical code
bhc --profile=numeric Main.hs

# Target WebAssembly
bhc --target=wasi Main.hs

# Target GPU (CUDA)
bhc --target=cuda --profile=numeric Main.hs
```

---

## Repository Structure

```
bhc/
├── crates/                    # Rust compiler implementation
│   ├── bhc/                   # Main CLI binary
│   ├── bhc-driver/            # Compilation orchestration
│   │
│   │   # Foundation crates
│   ├── bhc-span/              # Source locations
│   ├── bhc-arena/             # Memory arenas
│   ├── bhc-intern/            # String interning
│   ├── bhc-index/             # Index types
│   ├── bhc-data-structures/   # Shared data structures
│   ├── bhc-diagnostics/       # Error reporting
│   │
│   │   # Frontend
│   ├── bhc-lexer/             # Tokenization
│   ├── bhc-ast/               # Abstract syntax tree
│   ├── bhc-parser/            # Parsing
│   ├── bhc-types/             # Type representation
│   ├── bhc-typeck/            # Type inference & checking
│   │
│   │   # Middle-end
│   ├── bhc-hir/               # High-level IR
│   ├── bhc-lower/             # AST → HIR lowering
│   ├── bhc-core/              # Core IR + evaluator
│   ├── bhc-hir-to-core/       # HIR → Core lowering
│   ├── bhc-tensor-ir/         # Tensor IR (Numeric profile)
│   ├── bhc-loop-ir/           # Loop IR (vectorization)
│   │
│   │   # Backend
│   ├── bhc-target/            # Target specifications
│   ├── bhc-codegen/           # Native code generation (LLVM)
│   ├── bhc-gpu/               # GPU backends (CUDA/ROCm)
│   ├── bhc-wasm/              # WebAssembly backend
│   ├── bhc-linker/            # Linking
│   │
│   │   # Infrastructure
│   ├── bhc-session/           # Compilation session
│   ├── bhc-query/             # Incremental compilation
│   ├── bhc-package/           # Package management
│   ├── bhc-interface/         # Module interfaces
│   ├── bhc-ffi/               # FFI support
│   └── bhc-macros/            # Procedural macros
│
├── rts/                       # Runtime system (Rust)
│   ├── bhc-rts/               # Core runtime
│   └── bhc-rts-gc/            # Garbage collector
│
├── stdlib/                    # Standard library (Rust FFI support)
│   ├── bhc-prelude/           # Prelude primitives
│   ├── bhc-base/              # Base library (char, etc.)
│   ├── bhc-containers/        # Data structures
│   ├── bhc-text/              # Text/ByteString (SIMD)
│   ├── bhc-transformers/      # Monad transformers
│   ├── bhc-numeric/           # Numeric/SIMD/BLAS
│   ├── bhc-concurrent/        # Concurrency/STM
│   ├── bhc-system/            # System/IO/Process
│   └── bhc-utils/             # Time/Random/JSON
│
├── tools/                     # Additional tools
│   ├── bhci/                  # Interactive REPL
│   └── bhi/                   # IR inspector
│
└── tests/                     # Test suites
    ├── conformance/           # H26 conformance tests
    ├── benchmarks/            # Performance benchmarks
    └── integration/           # Integration tests
```

---

## CLI Tools

| Command | Description |
|---------|-------------|
| `bhc` | Compiler driver |
| `bhci` | Interactive REPL |
| `bhi` | IR inspector / kernel reports |

### Common Usage

```bash
# Compile to executable
bhc Main.hs -o main

# Check without generating code
bhc check Main.hs

# Run directly (via interpreter)
bhc run Main.hs

# Emit intermediate representations
bhc --dump-ir=core Main.hs
bhc --dump-ir=tensor Main.hs
bhc --dump-ir=loop Main.hs

# Kernel fusion report (Numeric profile)
bhc --profile=numeric --kernel-report Main.hs
```

---

## Runtime Profiles

Profiles define behavioral + performance contracts. Profiles are explicit and localizable (per package or per module).

| Profile | Use Case | Key Characteristics |
|---------|----------|---------------------|
| **default** | General Haskell | Lazy evaluation, GC managed |
| **server** | Web services, daemons | Structured concurrency, bounded latency, observability |
| **numeric** | ML, linear algebra, tensors | Strict-by-default, unboxed, fusion guaranteed, SIMD |
| **edge** | WASM, serverless | Minimal runtime footprint |
| **realtime** | Games, audio, robotics | Bounded GC pauses (<1ms), arena allocators |
| **embedded** | Microcontrollers | No GC, static allocation only |

### Profile Selection

```bash
# Command line
bhc --profile=numeric Main.hs

# Per-module pragma
{-# OPTIONS_BHC -profile=numeric #-}
module HotPath where
```

---

## Target Backends

| Target | Command | Status | Notes |
|--------|---------|--------|-------|
| **Native** | `bhc Main.hs` | 🔄 | LLVM backend, all profiles |
| **WASI/WASM** | `bhc --target=wasi Main.hs` | 🔄 | WebAssembly + WASI |
| **CUDA** | `bhc --target=cuda Main.hs` | 🔄 | NVIDIA GPU (PTX) |
| **ROCm** | `bhc --target=rocm Main.hs` | 🔄 | AMD GPU (AMDGCN) |
| **RISC-V** | `bhc --target=riscv64 Main.hs` | 🔄 | Via LLVM |

### Target + Profile Combinations

| Profile | Native | WASI | GPU |
|---------|--------|------|-----|
| default | ✅ | ✅ | ❌ |
| server | ✅ | 🟡 | ❌ |
| numeric | ✅ | ✅ | ✅ |
| edge | ✅ | ✅ | ❌ |
| realtime | ✅ | ❌ | ❌ |
| embedded | ✅ | ❌ | ❌ |

---

## Haskell Editions

BHC supports multiple Haskell editions for compatibility:

```bash
bhc --edition=Haskell2010 Main.hs   # Haskell 2010 standard
bhc --edition=GHC2021 Main.hs       # GHC2021 defaults
bhc --edition=GHC2024 Main.hs       # GHC2024 defaults
bhc --edition=H26 Main.hs           # Haskell 2026 (default)
```

---

## Key Technical Specifications

### Tensor IR (Numeric Profile)

The Tensor IR is the heart of BHC's numeric performance. Each tensor operation tracks:

| Property | Type | Description |
|----------|------|-------------|
| `dtype` | `DType` | Element type (Float32, Float64, etc.) |
| `shape` | `[Dim]` | Dimension sizes |
| `strides` | `[Stride]` | Byte strides per dimension |
| `layout` | `Layout` | Contiguous, Strided, or Tiled |
| `alias` | `Maybe BufferId` | Aliasing information |

### Fusion Guarantees

These patterns MUST fuse without intermediate allocation in Numeric profile:

```haskell
-- Pattern 1: map composition
map f (map g xs)           -- → map (f . g) xs

-- Pattern 2: zipWith with maps
zipWith f (map g a) (map h b)  -- → single traversal

-- Pattern 3: fold of map
sum (map f xs)             -- → single traversal

-- Pattern 4: strict fold of map
foldl' op z (map f xs)     -- → single traversal
```

Fusion failure in Numeric profile is a **compiler bug**.

### Memory Model

Three allocation regions:

| Region | Allocation | Deallocation | GC | Use Case |
|--------|------------|--------------|-----|----------|
| **Hot Arena** | Bump pointer O(1) | Bulk free at scope end | None | Kernel temporaries |
| **Pinned Heap** | malloc-style | Explicit/refcounted | Never moved | FFI, DMA, GPU |
| **General Heap** | GC-managed | Automatic | May move | Normal boxed data |

### Structured Concurrency (Server Profile)

Required primitives:

```haskell
-- Scope management
withScope :: (Scope -> IO a) -> IO a
withDeadline :: Duration -> (Scope -> IO a) -> IO (Maybe a)

-- Task management
spawn :: Scope -> IO a -> IO (Task a)
await :: Task a -> IO a
cancel :: Task a -> IO ()
poll :: Task a -> IO (Maybe a)

-- STM
atomically :: STM a -> IO a
newTVar :: a -> STM (TVar a)
readTVar :: TVar a -> STM a
writeTVar :: TVar a -> a -> STM ()
retry :: STM a
orElse :: STM a -> STM a -> STM a
```

### GPU Acceleration (Numeric Profile)

```haskell
{-# LANGUAGE BHC.TensorIR #-}
module Compute where

import BHC.Tensor

-- Matrix multiplication - automatically offloaded to GPU
matmul :: Matrix Double -> Matrix Double -> Matrix Double
matmul a b = T.contract a b

-- Operations fuse into GPU kernels
softmax :: Vector Double -> Vector Double
softmax v = T.map (/ total) exps
  where
    maxVal = T.maximum v
    exps = T.map (\x -> exp (x - maxVal)) v
    total = T.sum exps
```

---

## Development Guidelines

### Language

BHC is implemented in **Rust** with the standard library interface in **Haskell**.

### Core Principles

1. **Correctness first** — Semantic correctness is non-negotiable
2. **Predictable performance** — No hidden allocations or thunks in Numeric Profile
3. **Transparency** — Kernel reports, fusion diagnostics, allocation tracking
4. **Modularity** — Clean IR boundaries, pluggable backends

### Code Quality

- All code MUST pass `cargo clippy` and `cargo fmt`
- All public APIs MUST have documentation
- All new features MUST have tests
- Performance-critical code MUST have benchmarks

### Commit Messages

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `perf:` Performance improvements
- `refactor:` Code restructuring
- `docs:` Documentation
- `test:` Test additions/changes
- `chore:` Build, tooling, etc.

---

## Building

```bash
# Build all crates
cargo build

# Build release
cargo build --release

# Run tests
cargo test

# Run specific crate tests
cargo test -p bhc-parser

# Run benchmarks
cargo bench

# Build and run bhc
cargo run --bin bhc -- Main.hs
```

---

## Testing

### Test Categories

1. **Unit Tests** — Per-crate functionality
2. **Integration Tests** — End-to-end compilation
3. **Conformance Tests** — H26 specification compliance
4. **Benchmarks** — Performance regression detection

### Running Tests

```bash
# All tests
cargo test

# Conformance suite
cargo test -p bhc-conformance

# Benchmarks
cargo bench
```

---

## Implementation Roadmap

### Current Status: Beta (Active — Pandoc Compilation Target)

The compiler builds cleanly (33 crates, 0 errors) and compiles real Haskell programs to native executables via LLVM. **190 E2E tests** cover hello world through GADTs, monad transformers, records, user-defined typeclasses with dictionary passing, stock deriving (8 classes), and 30+ GHC extensions. 70 implementation milestones (E.1–E.70) completed. Separate compilation pipeline complete: `-c` mode, `.bhi` interface generation/consumption, `--odir`/`--hidir`/`--package-db` flags for hx package manager integration. The hx build pipeline is wired: `hx-bhc` generates correct BHC CLI flags, uses filesystem-based package DB, and maps standard Haskell packages to BHC builtins. The runtime system includes a work-stealing scheduler and STM support (both real, but currently exercised only from Rust — not wired to compiled Haskell). **Note (2026-07-02, see spec/BHC-REVIEW-0001 §5.1): the generational/incremental GC is a unit-tested module that is NOT wired to compiled code — `bhc_alloc` is a leak-allocator and the collector is a stub, so compiled programs leak all heap allocations. This is safe (nothing is freed, so nothing dangles) and fine for short-lived batch like Pandoc, but a real GC is deferred until a long-running target needs it.** Current focus: compiling Pandoc as the north-star integration target (see `.claude/TODO-pandoc.md`). Key remaining gaps: Core IR optimizer (simplifier with local+top-level transforms and case-of-case landed; pattern match compilation with decision trees and exhaustiveness checking landed; demand analysis + worker/wrapper landed; dictionary specialization landed), end-to-end Hackage package testing. WASM backend (update 2026-07-02, measured): produces valid binaries that run in wasmtime, at parity with native across the differential suite — 236/243 fixtures byte-identical (compute, stdout, closures, thunks, ADTs, typeclasses, fusion, host file IO). The old "binaries fail wasmtime validation" claim is stale. File IO is host-backed via WASI `path_open`/`fd_read`/`fd_write` (needs a `--dir` preopen; `generate_file_read_host`/`generate_file_write_host` in `bhc-wasm/src/wasi.rs`), with an in-memory table fallback for no-`--dir` round-trips; `lines`/`words` synthesized in `core_lower.rs`. Remaining WASM divergences are general stdlib coverage: `Handle` API + `System.Directory` unimplemented, plus the `show_types` native erased-type case and stdin fixtures. NOTE: native link needs staticlibs `libbhc_{rts,base,containers,text}.a` — build with `cargo build -p bhc -p bhc-rts -p bhc-base -p bhc-containers -p bhc-text` (a bare `--bin bhc` drops the .a files). GPU backend passes mock tests but requires CUDA hardware for real testing. REPL and tools compile but have stubbed evaluation.

### Phase 1: Core Compilation ✅ COMPLETE

**Goal:** Compile and run `main = putStrLn "Hello, World!"` to a native executable.

| Task | Status | Crate | Description |
|------|--------|-------|-------------|
| 1.1 LLVM Integration | 🟢 | bhc-codegen | inkwell/llvm-sys integrated, multi-target support |
| 1.2 Core → LLVM | 🟢 | bhc-codegen | 8,000+ lines: literals, functions, case, ADTs, closures |
| 1.3 RTS Bootstrap | 🟢 | bhc-rts | Entry points, allocation, GC roots, profile configs |
| 1.4 Basic GC | 🟢 | bhc-rts-gc | Generational collector (nursery/survivor/old) |
| 1.5 Linking | 🟢 | bhc-linker | Multi-platform (Unix/Windows/WASM), static/dynamic |
| 1.6 IO Primitives | 🟢 | bhc-rts | putStrLn, print, putChar via FFI |

**Exit Criteria:** ✅ `bhc Main.hs -o main && ./main` prints "Hello, World!"

### Phase 2: Language Completeness ✅ COMPLETE

**Goal:** Compile real Haskell programs (e.g., small utilities).

| Task | Status | Crate | Description |
|------|--------|-------|-------------|
| 2.1 Pattern Matching Codegen | 🟢 | bhc-codegen | Full ADT matching, nested patterns, tag dispatch |
| 2.2 Closures | 🟢 | bhc-codegen | Free variable capture, closure allocation/invocation |
| 2.3 Thunks & Laziness | 🟢 | bhc-rts | Thunk creation, forcing, blackhole detection |
| 2.4 Type Classes | 🟢 | bhc-typeck | Dictionary passing, default methods, superclass propagation |
| 2.5 Let/Where Bindings | 🟢 | bhc-codegen | Recursive and non-recursive, proper scoping |
| 2.6 Recursion | 🟢 | bhc-codegen | Mutual recursion, tail call optimization |
| 2.7 Prelude | 🟢 | stdlib | Full instances for Int/Float/Double/Char, FFI primitives in RTS |

**Exit Criteria:** ✅ Recursive Fibonacci compiles and runs correctly.

### Phase 3: Numeric Profile ✅ COMPLETE

**Goal:** Deliver promised numeric performance features.

| Task | Status | Crate | Description |
|------|--------|-------|-------------|
| 3.1 Core → Tensor IR | 🟢 | bhc-tensor-ir | Lower numeric Core to Tensor IR |
| 3.2 Fusion Passes | 🟢 | bhc-tensor-ir | All 4 guaranteed patterns per H26-SPEC |
| 3.3 Tensor → Loop IR | 🟢 | bhc-loop-ir | Lower Tensor IR to explicit loops |
| 3.4 Vectorization | 🟢 | bhc-loop-ir | SIMD auto-vectorization pass |
| 3.5 Parallelization | 🟢 | bhc-loop-ir | Parallel loop detection and codegen |
| 3.6 Loop → LLVM | 🟢 | bhc-codegen | Loop IR to LLVM IR lowering |
| 3.7 Hot Arena | 🟢 | bhc-rts-arena | Bump allocator, scope-based lifetime |
| 3.8 Pinned Buffers | 🟢 | bhc-rts-alloc | PinnedAllocator, PinnedBuffer, FFI API |
| 3.9 Kernel Reports | 🟢 | bhc-tensor-ir | Fusion report generation |

**Exit Criteria:** `sum (map (*2) [1..1000000])` fuses to single loop, runs 10x faster than interpreted. **⚠️ NOT MET on native (measured 2026-07-03):** the kernel report shows 0 fused/0 kernels for this program and numeric-profile timing equals default — bhc has no native list fusion (the Tensor/Loop fusion pipeline only feeds GPU/WASM + the report; native compiles Core→LLVM unfused, and `lower_module` recognizes 0 ops for list code). Making the numeric profile an honest native perf contract needs a Core→Core fusion pass for the 4 guaranteed patterns (+ enumFromTo elimination). See ROADMAP Phase 3 exit criteria.

### Phase 4: WASM Backend 🟡 70% COMPLETE

**Goal:** Compile to WebAssembly with WASI support.

| Task | Status | Crate | Description |
|------|--------|-------|-------------|
| 4.1 WASM Emitter | 🟡 | bhc-wasm | Binary emission exists but output fails wasmtime validation |
| 4.2 WASI Runtime | 🟢 | bhc-wasm | fd_write, proc_exit, print_i32, alloc, _start |
| 4.3 Loop IR Lowering | 🟢 | bhc-wasm | Complete statement/loop/op lowering to WASM |
| 4.4 Memory Model | 🟢 | bhc-wasm | LinearMemory, MemoryLayout, WasmArena |
| 4.5 Driver Integration | 🟢 | bhc-driver | Loop IR → WASM pipeline wiring |

**Exit Criteria:** `bhc --target=wasi Main.hs -o app.wasm && wasmtime app.wasm` works.

**Notes:** All 6 WASM E2E tests fail with "WebAssembly translation error". The emitter produces output but the WASM binary format is not valid.

### Phase 5: Server Profile 🟡 RTS-COMPLETE, NOT WIRED to compiled code

**Goal:** Structured concurrency with work-stealing scheduler.

> The 🟢 rows below are the **Rust RTS** modules (real, tested). No compiled Haskell reaches them — `spawn`/`await`/`withScope`/`atomically` are unwired stdlib signatures, 0 concurrency E2E fixtures (spec/BHC-REVIEW-0001 §5.2).

| Task | Status | Crate | Description |
|------|--------|-------|-------------|
| 5.1 Task Scheduler | 🟢 | bhc-rts-scheduler | Work-stealing with crossbeam deques, 24 tests |
| 5.2 Scope Primitives | 🟢 | bhc-concurrent | withScope, spawn, await, nested scopes |
| 5.3 Cancellation | 🟢 | bhc-concurrent | Cooperative cancellation, <1ms propagation |
| 5.4 STM Runtime | 🟢 | bhc-concurrent | TVar, atomically, retry, orElse, TMVar, TQueue (30 tests) |
| 5.5 Deadlines | 🟢 | bhc-concurrent | withDeadline, timeout, deadline propagation |
| 5.6 Observability | 🟢 | bhc-rts-scheduler | TraceEvent system with 10+ event types |

**Exit Criteria:** the 11 M5 tests pass **from Rust**; the compiled-Haskell exit criterion (a concurrent program that compiles and runs) is NOT met — wiring pending.

### Phase 6: GPU Backend 🟡 80% COMPLETE

**Goal:** Offload numeric kernels to GPU.

| Task | Status | Crate | Description |
|------|--------|-------|-------------|
| 6.1 PTX Codegen | 🟢 | bhc-gpu | NVIDIA PTX emission (Map, ZipWith, Reduce with parallel reduction) |
| 6.2 AMDGCN Codegen | 🟡 | bhc-gpu | AMD AMDGCN emission (structure complete, needs testing) |
| 6.3 Device Memory | 🟢 | bhc-gpu | Host/device transfer management via CUDA FFI |
| 6.4 Kernel Launch | 🟢 | bhc-gpu | GPU kernel invocation with dynamic CUDA loading |
| 6.5 Tensor → GPU | 🟢 | bhc-gpu | Lower Tensor IR to GPU kernels with caching |

**Exit Criteria:** Matrix multiplication runs on GPU, 100x faster than CPU for large matrices.

**Notes:** 2/2 GPU mock tests pass (PTX validation). End-to-end testing requires CUDA hardware.

### Phase 7: Advanced Profiles 🟡 IN PROGRESS (GC not wired)

**Goal:** Realtime and Embedded profiles.

> 7.1 GC is a unit-tested module NOT on the compiled-code path (`bhc_alloc` leaks; §5.1). Arena (7.2) and no-GC embedded (7.3) are real allocation paths.

| Task | Status | Crate | Description |
|------|--------|-------|-------------|
| 7.1 Incremental GC | 🟢 module only, not wired | bhc-rts-gc | Pause measurement, tri-color marking, SATB barriers |
| 7.2 Arena per-frame | 🟢 | bhc-rts-arena | FrameArena with begin/end lifecycle, double buffering |
| 7.3 No-GC Mode | 🟢 | bhc-rts-alloc | StaticAllocator, BoundedAllocator, Embedded profile |
| 7.4 Bare Metal | 🟡 | bhc-codegen | No-OS code generation (deferred - needs LLVM target work) |

**Exit Criteria:** Game loop demo with <1ms GC pauses.

**Notes:** Realtime and Embedded profiles added to RTS. Pause tracking with P99 percentiles, threshold violations, and ring buffer history. Incremental marking supports time-budgeted work increments (default 500μs).

### Phase 8: Ecosystem 🟡 60% COMPLETE

**Goal:** Production-ready tooling.

| Task | Status | Crate | Description |
|------|--------|-------|-------------|
| 8.1 REPL | 🟡 | bhci | Compiles, but evaluation is stubbed |
| 8.2 IR Inspector | 🟡 | bhi | Compiles, needs integration testing |
| 8.3 Package Manager | 🟡 | hx + bhc-interface | hx is the package manager; BHC side (-c, .bhi) + hx-bhc wiring complete; end-to-end testing needed |
| 8.4 LSP Server | 🟡 | bhc-lsp | Code exists, needs testing |
| 8.5 Documentation | 🟡 | - | User docs exist, API docs incomplete |

**Exit Criteria:** Developers can build, test, and deploy BHC projects.

### Phase 9: Real-World Haskell Compatibility 🟡 70% COMPLETE

**Goal:** Compile real-world Haskell projects (Pandoc as north-star target).

| Task | Status | Description |
|------|--------|-------------|
| 9.1 GHC Extensions | 🟢 | 30+ extensions: OverloadedStrings, GADTs, FlexibleInstances, MultiParamTypeClasses, FunctionalDependencies, ScopedTypeVariables, TypeOperators, GeneralizedNewtypeDeriving, DeriveGeneric, DeriveFunctor/Foldable/Traversable, DeriveAnyClass, StandaloneDeriving, PatternSynonyms, ViewPatterns, RecordWildCards, etc. |
| 9.2 Typeclass System | 🟢 | User-defined typeclasses with dictionary passing, higher-kinded, default methods, superclasses, DeriveAnyClass |
| 9.3 Record Syntax | 🟢 | Named fields, accessors, construction, update, RecordWildCards, NamedFieldPuns |
| 9.4 Stock Deriving | 🟢 | 8 classes: Eq, Show, Ord, Enum, Bounded, Functor, Foldable, Traversable + Generic stubs |
| 9.5 Cross-Transformer Codegen | 🟢 | All combinations: StateT/ReaderT, ExceptT/StateT+ReaderT, WriterT/StateT+ReaderT |
| 9.6 Layout Rule | 🟢 | Haskell 2010 indentation-based layout (where, let, do, case/of, guards, class/instance) |
| 9.7 Core IR Optimizer | 🟡 | Core simplifier (E.68-E.69): constant folding, beta reduction, case-of-known-constructor, case-of-case, local+top-level dead/inline; top-level inlining (cheap-only, protected names skipped), export-aware dead elimination; pattern match compilation (E.70): Augustsson/Sestoft decision trees, exhaustiveness warnings, overlap detection; demand analysis + worker/wrapper (O.3): boolean-tree strictness signatures, fixpoint iteration for recursive groups, worker/wrapper split for strict args, gated on lazy profiles; dictionary specialization (O.4): direct method selection on known dictionaries, cleanup simplifier pass |
| 9.8 Package System | 🟡 | Separate compilation pipeline complete (-c, .bhi, --package-db); hx build pipeline wired (correct flags, filesystem DB, builtin mapping); end-to-end Hackage testing needed |
| 9.9 CPP Preprocessing | 🟢 | Built-in Rust preprocessor: `#ifdef`/`#if`/`#elif`/`#else`/`#endif`/`#define`/`#undef`, expression evaluator, macro expansion, predefined platform/version macros |
| 9.10 Type Families | 🟢 | Standalone open/closed type families, type instances, associated type families with reduction; standalone data families with data instances |

**Exit Criteria:** `bhc check` succeeds on Pandoc source files (excluding Template Haskell). **Status (2026-07-23): 112 of 221 library modules pass** (up from ~10); remaining tail is deep typeck work. See `.claude/TODO-pandoc-check.md`.

**Notes:** 190 E2E tests passing across 70 milestones (E.1–E.70); workspace `cargo test --all-features` 2756/0. See `.claude/TODO-pandoc-check.md` for the current Pandoc grind.

---

### Roadmap Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 | Not started |
| 🟡 | Partial / In progress |
| 🟢 | Complete |

### Priority Order

1. **Phase 1** — Without native codegen, nothing else matters ✅
2. **Phase 2** — Language features needed for real programs ✅
3. **Phase 3** — Numeric profile is our differentiator 🟡 (IR built; native fusion NOT met — the differentiator is unvalidated)
4. **Phase 9** — Real-world Haskell compatibility (**current focus**; Pandoc 112/221)
5. **Phase 4** — WASM opens new deployment targets 🟢 (~95%, runs in wasmtime)
6. **Phase 5** — Server profile for production services 🟡 (RTS done; not wired to compiled code)
7. **Phase 6** — GPU for competitive numeric performance
8. **Phase 7** — Advanced profiles for specialized use cases
9. **Phase 8** — Polish and ecosystem

---

## References

- Website: https://bhc.raskell.io
- Repository: https://github.com/raskell-io/bhc
- See `.claude/rules/` for detailed coding guidelines
