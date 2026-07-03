# BHC Implementation Roadmap

This document provides a detailed implementation plan to deliver all features promised on [bhc.raskell.io](https://bhc.raskell.io).

## Current Status

**Beta** — The compiler builds cleanly and compiles real Haskell programs to native executables. E2E tests confirm native compilation works for hello world, arithmetic, fibonacci, IO sequencing, and more. **WASM update (2026-07-02, measured):** the WASM/WASI backend produces valid binaries that run in wasmtime and is at parity with native across the differential suite — **236 of 243 fixtures produce byte-identical output on both backends** (the harness now stages fixture data files and runs `wasmtime --dir=.`). File IO is **host-backed** via WASI `path_open`/`fd_read`/`fd_write` (needs a `--dir` preopen at runtime): `readFile` reads real files, `writeFile` persists to disk, with an in-memory table backing round-trips when no `--dir` is given; `lines`/`words` are synthesized so "read a file, process its lines, write it back" works. Remaining WASM divergences are **general stdlib coverage**, not filesystem plumbing: the `Handle` API (`openFile`/`hGetLine`/`hClose`) and `System.Directory` (`doesFileExist`) unimplemented, plus the long-standing `show_types` native erased-type case and stdin fixtures the harness can't feed. The old "WASM binaries fail wasmtime validation" claim below is stale. GPU backend remains mock-validated only.

| Component | Status | Test Evidence |
|-----------|--------|---------------|
| Parser/Lexer | ✅ Complete | 76 unit tests pass |
| Type Checker | ✅ Complete | Inference + type classes functional |
| HIR Lowering | ✅ Complete | Used in all compilation paths |
| Core IR | ✅ Complete | 30 interpreter tests pass; LLVM codegen works |
| Tensor IR | ✅ Complete | Lowering, fusion, all 4 patterns |
| Loop IR | ✅ Complete | Vectorization, parallelization |
| Native Codegen | ✅ Complete | 6/6 E2E tests pass (hello, arithmetic, fibonacci, IO) |
| WASM Codegen | 🟢 ~95% | Valid binaries; run in wasmtime; 232/243 differential fixtures byte-identical to native (2026-07-02). Gap: file IO not host-backed (in-memory shim) |
| GPU Codegen | 🟡 80% | PTX mock validation passes; requires CUDA hardware for real test |
| Runtime | 🟡 Partial | Arena + scheduler + STM + thunks work. **GC is NOT live for compiled code** — `bhc_alloc` is a leak-allocator and the collector is a stub; the generational/incremental GC exists only as an isolated, unit-tested module (see spec/BHC-REVIEW-0001 §5.1). Safe-but-leaks; adequate for short-lived batch, not long-running programs. |
| REPL (bhci) | 🟡 Compiles | Builds but evaluation is stubbed (returns placeholder values) |
| Package Manager | 🟡 Partial | Code exists, some test imports broken |
| LSP Server | 🟡 Exists | Code present, not independently verified |
| Documentation | 🟡 Partial | User docs exist, API docs incomplete |

### Test Summary (Jan 29, 2026 — STALE; see 2026-07-02 differential note below)

| Suite | Passed | Failed | Ignored | Notes |
|-------|--------|--------|---------|-------|
| Workspace unit tests | 217 | 9 | 0 | 9 failures are interpreter IO capture |
| E2E Native | 6 | 0 | 1 | Numeric profile test ignored |
| E2E WASM | 0 | 6 | 1 | STALE — "WASM translation error" no longer reproduces |
| E2E GPU | 2 | 0 | 0 | Mock mode only |
| **Total** | **225** | **15** | **2** | |

> **Differential sweep (2026-07-02, `crates/bhc-e2e-tests/differential.py`, 243 fixtures):**
> native vs WASM — **232 agree (byte-identical), 11 divergences.** None of the divergences
> are WASM failing where native succeeds on compute/stdout. The divergences are: file-IO
> fixtures (harness doesn't stage the input file, so native errors while WASM silently returns
> empty — see the file-IO gap above) and 1 fixture wrong on both. The old "E2E WASM 0/6" row
> is obsolete. NOTE: WASM file IO is not host-backed — do not read "232 agree" as full IO parity.

---

## Phase 1: Native Hello World ✅ COMPLETE

**Objective:** `bhc Main.hs -o main && ./main` prints "Hello, World!"

This is the critical path. Everything else depends on native code generation working.

### 1.1 LLVM Integration ✅

**Crate:** `bhc-codegen`
**Dependency:** [inkwell](https://crates.io/crates/inkwell) (safe LLVM bindings)

Tasks:
- [x] Add `inkwell` dependency to `bhc-codegen/Cargo.toml`
- [x] Create `LlvmContext` wrapping inkwell's `Context`
- [x] Create `LlvmModule` wrapping inkwell's `Module`
- [x] Implement `CodegenBackend` trait with real LLVM operations
- [x] Remove placeholder implementations
- [x] Add target triple detection from `bhc-target`
- [x] Test: Create and verify a simple LLVM module

### 1.2 Core IR to LLVM ✅

**Crate:** `bhc-codegen`

Tasks:
- [x] Define LLVM type mappings for Core IR types
- [x] Implement `Lit` (literal) codegen
- [x] Implement `Var` (variable) codegen
- [x] Implement `App` (application) codegen
- [x] Implement `Lam` (lambda) codegen
- [x] Implement `Let` codegen
- [x] Implement `Case` codegen (basic)
- [x] Test: Compile `main = 1 + 2` to working executable

### 1.3 Minimal Runtime System ✅

**Crate:** `bhc-rts`

Tasks:
- [x] Define object header layout
- [x] Define info table structure
- [x] Implement `bhc_alloc(size: usize) -> *mut u8`
- [x] Implement `bhc_init()` - runtime initialization
- [x] Implement `bhc_exit(code: i32)` - clean shutdown
- [x] Create RTS static library for linking
- [x] Test: Link a trivial program with RTS

### 1.4 Basic GC 🟡 (module only — NOT wired to compiled code)

**Crate:** `bhc-rts-gc`

> **Correction (2026-07-02, spec/BHC-REVIEW-0001 §5.1):** The tasks below are complete *as an isolated, unit-tested module*, but the collector is **not on the compiled-code path**. `bhc_alloc` (`rts/bhc-rts/src/ffi.rs`) is a raw malloc; `major_collect`/`minor_collect` are stubs that free nothing; native codegen emits no roots and no info tables. Compiled programs leak all heap allocations. "Root set tracking" / "GC trigger" / "live objects survive" hold only inside the crate's own tests, not for real programs.

Tasks (module-level, in isolation):
- [x] Implement root set tracking *(module type only; not populated by codegen)*
- [x] Implement mark phase *(over an abstract GcPtr graph in tests)*
- [x] Implement sweep phase *(over an abstract GcPtr graph in tests)*
- [x] Add GC trigger (allocation threshold) *(not wired: `bhc_alloc` never triggers it)*
- [x] Test: Allocate objects, trigger GC, verify live objects survive *(unit test only)*
- [ ] **Wire to compiled heap** (info tables + rooting + real collect) — deferred; see review §5.1

### 1.5 Linker Integration ✅

**Crate:** `bhc-linker`

Tasks:
- [x] Detect system linker (ld, lld, link.exe)
- [x] Generate object file from LLVM module
- [x] Link object file with RTS library
- [x] Handle platform-specific linking flags
- [x] Test: Full compile-link pipeline produces working executable

### 1.6 IO Primitives ✅

**Crate:** `bhc-rts`

Tasks:
- [x] Implement `bhc_print_int_ln(i: i64)` - print integer with newline
- [x] Implement `bhc_print_double_ln(d: f64)` - print double with newline
- [x] Implement `bhc_print_string_ln(ptr: *const u8, len: usize)` - print string with newline
- [x] Wire up Haskell `print` to RTS functions
- [x] Test: `main = print 42` program works

### Phase 1 Exit Criteria ✅

```bash
$ cat Main.hs
main = print 42

$ bhc run Main.hs
42
```

**Completed!** Verified by E2E test `test_tier1_arithmetic_native` (Jan 29, 2026).

**Note:** `main = putStrLn "Hello, World!"` also compiles and runs correctly via LLVM native codegen. String IO works in the codegen path but not yet in the interpreter path (display capture issue).

---

## Phase 2: Language Completeness ✅ COMPLETE

**Objective:** Compile and run real Haskell programs.

### 2.1 Pattern Matching Codegen ✅

**Crate:** `bhc-codegen`
**Location:** `lower.rs` lines 6942-7850

Tasks:
- [x] Implement constructor pattern matching (`lower_case_datacon()`)
- [x] Implement nested patterns (field extraction, decision trees)
- [x] Implement guards
- [x] Implement as-patterns (`x@(Cons a b)`)
- [x] Implement wildcard patterns
- [x] Implement literal patterns (`lower_case_literal_int/float/string()`)
- [x] Test: Pattern matching on Maybe, Either, lists

### 2.2 Closures ✅

**Crate:** `bhc-codegen`, `bhc-rts`
**Location:** `lower.rs` lines 4147-5100

Tasks:
- [x] Define closure object layout (`{ fn_ptr, env_size, env[] }`)
- [x] Implement closure allocation (`alloc_closure()`)
- [x] Implement closure entry code
- [x] Implement free variable analysis (`free_vars()`, `collect_free_vars()`)
- [x] Generate closure-creating code for lambdas (`lower_lambda()`)
- [x] Test: Higher-order functions (`map`, `filter`)

### 2.3 Thunks & Laziness ✅

**Crate:** `bhc-rts`, `bhc-codegen`
**Location:** `lower.rs` lines 4426-4584

Tasks:
- [x] Define thunk object layout (`{ tag, eval_fn, env_size, env[] }`)
- [x] Implement thunk evaluation (`build_force()` → `bhc_force()`)
- [x] Implement thunk tag checking (`bhc_is_thunk()`)
- [x] Implement indirection handling
- [x] Generate thunk-creating code (`alloc_thunk()`, `lower_lazy()`)
- [x] Test: Lazy infinite list `[1..]`

### 2.4 Type Classes ✅

**Crate:** `bhc-typeck`, `bhc-codegen`
**Location:** `context.rs` lines 206-327, `env.rs` lines 287-306

Tasks:
- [x] Implement instance resolution algorithm
- [x] Implement dictionary passing via field selectors (`$sel_N`)
- [x] Implement dictionary construction for instances
- [x] Handle superclass constraints (e.g., `Ord a` implies `Eq a`)
- [x] Test: `Eq`, `Ord`, `Show` instances for primitives

### 2.5 Let/Where Bindings ✅

**Crate:** `bhc-codegen`
**Location:** `lower.rs` lines 6620-6700

Tasks:
- [x] Implement non-recursive let
- [x] Implement recursive let (letrec) - lifted to top-level functions
- [x] Implement where clauses (desugar to let)
- [x] Test: Mutual recursion in let

### 2.6 Recursion & Tail Calls ✅

**Crate:** `bhc-codegen`
**Location:** `lower.rs` lines 96-112

Tasks:
- [x] Detect tail call positions (`in_tail_position` tracking)
- [x] Implement tail call optimization (`call.set_tail_call(true)`)
- [x] Implement self-recursive tail calls
- [x] Test: `factorial 1000000` without stack overflow

### 2.7 Prelude Bootstrap ✅

**Crate:** `stdlib/bhc-prelude`
**Location:** `hs/BHC/Prelude.hs` (650+ lines)

Tasks:
- [x] Compile basic list functions (`map`, `filter`, `foldr`, `foldl`, 30+ functions)
- [x] Compile Maybe/Either functions
- [x] Compile numeric operations (100+ FFI primitives)
- [x] Implement 26 type classes (Eq, Ord, Num, Functor, Monad, etc.)
- [x] Test: Compile program using Prelude

### Phase 2 Exit Criteria ✅

```haskell
-- Fibonacci
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2)

main = print (fib 30)
```

**Completed!** Commit `745bbac` (Jan 26, 2026)

**Bug Fix (Jan 28, 2026):** Fixed two pattern matching bugs in `lower.rs` (commit `7db9592`):
1. Switch default block was using error fallback instead of user's wildcard
2. Trivial case optimization was bypassing literal pattern matching

---

## Phase 3: Numeric Profile ✅ COMPLETE

**Objective:** Deliver promised numeric performance: fusion, SIMD, parallelism.

### 3.1 Core to Tensor IR Lowering ✅

**Crate:** `bhc-tensor-ir`
**Location:** `lower.rs` (1,259 lines)

Tasks:
- [x] Identify numeric operations via `BuiltinTable` (12+ operations)
- [x] Lower array/vector operations to Tensor IR
- [x] Track shape information via `TensorMeta`
- [x] Track element types
- [x] Test: Lower `map (*2) xs` to TensorMap

### 3.2 Fusion Implementation ✅

**Crate:** `bhc-tensor-ir`
**Location:** `fusion.rs` (2,715 lines)

Tasks:
- [x] Implement Pattern 1: `map f (map g x)` → single traversal (MapMap)
- [x] Implement Pattern 2: `zipWith f (map g a) (map h b)` → single traversal (ZipWithMaps)
- [x] Implement Pattern 3: `sum (map f x)` → single traversal (ReduceMap)
- [x] Implement Pattern 4: `foldl' op z (map f x)` → single traversal (FoldMap)
- [x] Add fusion verification via reference counting
- [x] Generate fusion report (`generate_kernel_report()` line 1823)
- [x] Test: Verify all 4 guaranteed patterns fuse

### 3.3 Tensor to Loop IR ✅

**Crate:** `bhc-loop-ir`
**Location:** `lower.rs` (500+ lines)

Tasks:
- [x] Generate explicit loop nests from Tensor ops (`lower_kernel()`)
- [x] Track loop bounds from shapes
- [x] Generate index calculations from strides
- [x] Handle broadcasting
- [x] Test: TensorMap becomes `for` loop

### 3.4 SIMD Vectorization ✅

**Crate:** `bhc-loop-ir`
**Location:** `vectorize.rs` (600+ lines)

Tasks:
- [x] Identify vectorizable loops (`VectorizePass`)
- [x] Compute vector width from target (`VectorizeConfig`)
- [x] Generate vector types (`LoopType::Vector(ScalarType, width)`)
- [x] Generate vector load/store
- [x] Generate vector operations with FMA detection
- [x] Handle loop remainders
- [x] Test: `sum xs` uses SIMD

### 3.5 Parallel Loop Codegen ✅

**Crate:** `bhc-loop-ir`
**Location:** `parallel.rs` (400+ lines)

Tasks:
- [x] Identify parallelizable loops (`ParallelPass`)
- [x] Generate parallel loop structure with 3 strategies (Static/Dynamic/Guided)
- [x] Integrate with RTS thread pool
- [x] Handle reduction across threads
- [x] Test: Parallel map scales with cores

### 3.6 Loop IR to LLVM ✅

**Crate:** `bhc-codegen`
**Location:** `llvm/loop_lower.rs` (74KB)

Tasks:
- [x] Lower Loop IR to LLVM IR (`LoopLowering` struct)
- [x] Emit LLVM vector intrinsics (fabs, sqrt, floor, ceil, FMA)
- [x] Use LLVM's loop optimizations
- [x] Test: Generated assembly contains SIMD

### 3.7 Hot Arena Allocator ✅

**Crate:** `bhc-rts-arena`
**Location:** `lib.rs` (400+ lines)

Tasks:
- [x] Implement arena allocation (`HotArena`, bump pointer O(1))
- [x] Implement bulk free (scope-based lifetime)
- [x] Support alignment (16/32/64 byte for SIMD)
- [x] Test: Kernel temporaries use arena

### 3.8 Pinned Buffers ✅

**Crate:** `bhc-rts-alloc`
**Location:** `lib.rs`

Tasks:
- [x] Implement pinned allocation (`PinnedAllocator`)
- [x] Track pinned objects separately from GC heap
- [x] Implement reference counting for pinned
- [x] Test: FFI buffer survives GC

### Phase 3 Exit Criteria — 🔴 NOT MET on native (corrected 2026-07-03)

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
main = print $ sum $ map (*2) [1..10000000]
```

**Measured reality (2026-07-03):** for this exact program the numeric profile does
NOT fuse on the native target:
- `--kernel-report` shows **0 fused / 0 loops / 0 kernels** (empty).
- Runtime is **identical to the default profile** (`sum (map (*2) (map (+1) [1..20M]))`
  takes ~7 s under both — a fused strict loop would take tens of ms).
- Root cause: (1) `bhc_tensor_ir::lower::lower_module` recognizes **0 ops** for
  standard list programs (it only lowers actual tensor ops), and (2) even when it
  does, the **native codegen ignores the fused Loop IR** — the Tensor/Loop pipeline
  in `bhc-driver` feeds only GPU (`tensor_kernels_for_gpu`) and WASM
  (`loop_irs_for_wasm`) + the kernel report; native still compiles Core→LLVM
  unfused. bhc has **no list fusion in the Core simplifier** at all.

So the numeric profile is currently a behavioral/config selector without the
native fusion performance contract it advertises. The prior "✅ Completed,
commit 312f08c" claim was not validated against a run.

**Attempted, and what it revealed (2026-07-03).** A Core→Core fusion pass was
built in `crates/bhc-core/src/simplify/fuse.rs` (gated on `fuse_lists`, numeric
only), and it exposed that a fusion pass *alone cannot make this honest*:

- **Core erases types.** By the time the simplifier runs, a lambda's type is
  `Fun(Error, Error)` — the element/result types are gone. So type-dependent
  patterns cannot fire: `sum (map f xs)` → `foldl'` needs to know the accumulator
  is `Int` (to emit `0`/`+`), and that information is not there. Only the
  type-agnostic `map f (map g xs)` → `map (f∘g) xs` rewrite fires.
- **Codegen boxes every `Int`.** Even when a rewrite fires, the flagship
  `sum (map f [1..N])` is unchanged (~6–11 s at `-O2` and `-O3`, both profiles):
  per-element boxing, not intermediate lists, dominates.

So an honest native numeric perf contract is blocked on two deeper properties —
**Core type preservation** (typed Core IR, per `rules/007-ir-design.md`) and
**unboxed numeric codegen** — not on the fusion pass. The `map/map` and `sum/map`
rewrites are retained as correct scaffolding (`map/map` fires; `sum/map` is inert
until Core preserves types); do not attribute any speedup to them without an
isolated numeric-with-fusion vs numeric-without-fusion measurement.

---

## Phase 4: WASM Backend 🟡 70% COMPLETE

**Objective:** `bhc --target=wasi Main.hs` produces working WebAssembly.

### 4.1 WASM Emitter ✅

**Crate:** `bhc-wasm`
**Location:** `codegen/mod.rs` (1,656 lines)

Tasks:
- [x] Emit WASM binary format (all 11 sections, LEB128 encoding)
- [x] Map types to WASM types (i32, i64, f32, f64, v128)
- [x] Generate WASM functions (`WasmFunc`, `WasmFuncType`)
- [x] Handle indirect calls (`CallIndirect` instruction)
- [x] SIMD128 support (`codegen/simd.rs`)
- [x] WAT text generation
- [x] Test: Valid WASM binary output

### 4.2 WASI Runtime Integration ✅

**Crate:** `bhc-wasm`
**Location:** `wasi.rs` (800+ lines)

Tasks:
- [x] Import WASI functions (fd_write, fd_read, proc_exit, args_*, environ_*)
- [x] Map IO primitives to WASI calls (`generate_print_i32()`, `generate_print_str()`)
- [x] Bump allocator (`generate_alloc_function()`)
- [x] Handle command-line arguments (`generate_init_args()`, `generate_get_argc()`, `generate_get_argv()`)
- [x] Handle environment variables (`generate_init_environ()`, `generate_getenv()`)
- [x] Test: Basic print works

### 4.3 Edge Profile RTS ✅

**Crate:** `bhc-wasm`
**Location:** `runtime/mod.rs` (520 lines), `runtime/gc.rs` (625 lines)

Tasks:
- [x] Configuration for minimal RTS (`RuntimeConfig::edge()`)
- [x] Memory layout definition (`MemoryLayout`)
- [x] Arena allocator for WASM (`WasmArena`)
- [x] Full GC within linear memory (`generate_gc_*` functions in gc.rs)
- [x] Code size verification test (`test_runtime_code_size_under_100kb`)
- [x] Test: Runtime < 100KB (17 functions, includes GC + WASI)

### 4.4 Driver Integration ✅

**Crate:** `bhc-driver`, `bhc` (CLI)
**Location:** `lib.rs` lines 467-537, 1109-1115; `main.rs`

Tasks:
- [x] Add `--target=wasi` flag handling (`is_wasm_target()` at line 1109)
- [x] Wire WASM backend into compilation pipeline (lines 467-537)
- [x] Register wasm32-wasi target (detected via "wasm" in target triple)
- [x] Generate `.wasm` output files (`write_wasm()` at line 525-527)
- [x] Add `--target` and `--emit` CLI flags to bhc binary (commit `7db9592`)
- [ ] Test: End-to-end compilation with wasmtime

### Phase 4 Exit Criteria

```bash
$ bhc --target=wasi Main.hs -o app.wasm
$ wasmtime app.wasm
Hello, World!
```

**LLVM 21 Support:** Upgraded inkwell from 0.5 (LLVM 18) to 0.8 (LLVM 21). Commit `db4368f`.

**Status (Jan 29, 2026 — STALE, superseded 2026-07-02):** ~~All 6 WASM E2E tests fail with "WebAssembly translation error"~~. This no longer reproduces. As of 2026-07-02 the WASM binaries are valid, run in wasmtime, and match native output on 232/243 differential fixtures (compute, stdout, closures, thunks, ADTs, typeclasses, fusion — not just numeric kernels). The validation issue was resolved (type-index/alignment encoding, see the "Recent Fixes" section).

**Remaining effort (2026-07-02):** (1) ~~Host-backed file IO~~ **DONE** — `readFile`/`writeFile`/`readFile'` go through WASI `path_open`/`fd_read`/`fd_write` (fd-3 preopen; `generate_file_read_host`/`generate_file_write_host` in `wasi.rs`), with the in-memory table as a no-`--dir` fallback. differential.py stages fixture data files + runs `wasmtime --dir=.`. (2) ~~`lines`/`words`~~ **DONE** — synthesized in `core_lower.rs` `build_list_fn` (break/dropWhile/isSpace-based; arg coerced via `emit_ensure_charlist`); flipped file_stats/file_reverse to agree. (3) TODO: `Handle` API (`openFile`/`hGetLine`/`hClose`) and `System.Directory` (`doesFileExist`/`doesDirectoryExist`) — larger, still unimplemented on WASM (fixtures handle_io/system_ops).

---

## Phase 5: Server Profile ✅ COMPLETE

**Objective:** Structured concurrency with proper cancellation.

### 5.1 Task Scheduler ✅

**Crate:** `bhc-rts-scheduler`
**Location:** `lib.rs` (1,459 lines)

Tasks:
- [x] Implement work-stealing deque (crossbeam)
- [x] Implement worker threads (configurable count)
- [x] Implement task spawning
- [x] Implement task completion with statistics
- [x] Test: 15 tests pass

### 5.2 Scope & Task Primitives ✅

**Crate:** `bhc-rts-scheduler`

Tasks:
- [x] Implement `Scope` type
- [x] Implement `with_scope()` (structured concurrency)
- [x] Implement `spawn()` within scope
- [x] Implement `await()` (blocking and non-blocking)
- [x] Test: Concurrent tasks complete within scope

### 5.3 Cancellation ✅

**Crate:** `bhc-rts-scheduler`
**Location:** lines 325-361

Tasks:
- [x] Implement cancellation tokens (thread-local flag)
- [x] Implement `cancel()` method
- [x] Implement cancellation propagation to children
- [x] Implement `check_cancelled()` cooperative checking
- [x] Test: Cancelled task stops

### 5.4 STM ✅

**Crate:** `bhc-concurrent`
**Location:** `stdlib/bhc-concurrent/src/stm.rs` (971 lines)

Tasks:
- [x] Implement `TVar` type with atomic versioning (lines 84-179)
- [x] Implement `atomically()` with retry/conflict handling (lines 393-460)
- [x] SATB write barriers
- [x] Implement `retry` primitive (lines 480-482)
- [x] Implement `orElse` combinator (lines 505-514)
- [x] Implement conflict detection (validation in `Transaction::commit()`)
- [x] Tests: 13 tests including bank transfer, producer-consumer (lines 782-970)

### 5.5 Deadlines ✅

**Crate:** `bhc-rts-scheduler`
**Location:** lines 1057-1109

Tasks:
- [x] Implement `with_deadline(duration, closure)`
- [x] Implement deadline propagation (timer thread)
- [x] Test: Operation times out

### Phase 5 Exit Criteria

```haskell
main = withScope $ \scope -> do
  t1 <- spawn scope $ do
    threadDelay 1000000
    return 1
  t2 <- spawn scope $ do
    threadDelay 500000
    return 2
  r1 <- await t1
  r2 <- await t2
  print (r1 + r2)
```

**Blockers:** None - core functionality complete.

**Remaining effort:** ~1 week (observability hooks, integration testing)

---

## Phase 6: GPU Backend 🟡 95% COMPLETE

**Objective:** Tensor operations run on GPU.

### 6.1 PTX Codegen ✅

**Crate:** `bhc-gpu`
**Location:** `codegen/ptx.rs` (1,168 lines)

Tasks:
- [x] PTX module header generation
- [x] Kernel entry point signatures
- [x] Parameter marshalling
- [x] Type mapping (`dtype_to_gpu_type`)
- [x] Loop nest code generation (`generate_loop_nest()`)
- [x] Map/ZipWith/Reduce operations
- [x] Parallel reduction with shared memory
- [ ] Test: Simple kernel compiles (requires CUDA)

### 6.2 AMDGCN Codegen ✅

**Crate:** `bhc-gpu`
**Location:** `codegen/amdgcn.rs` (580 lines)

Tasks:
- [x] AMDGCN module header
- [x] Kernel entry generation
- [x] Parameter handling
- [x] Loop nest code generation (`generate_loop_nest_amd()`)
- [x] Unary/binary operations
- [ ] Test: Simple kernel compiles (requires ROCm)

### 6.3 Device Memory Management ✅

**Crate:** `bhc-gpu`
**Location:** `memory.rs`

Tasks:
- [x] Implement device allocation (`DeviceBuffer<T>`)
- [x] Pool-based memory management
- [x] Alignment tracking
- [x] Safety checks for bounds

### 6.4 Host-Device Transfers ✅

**Crate:** `bhc-gpu`
**Location:** `transfer.rs`

Tasks:
- [x] Implement host→device transfer
- [x] Implement device→host transfer
- [x] Device-to-device copy
- [x] Async transfer support via streams
- [x] Test: Data flows to/from GPU

### 6.5 Kernel Launch ✅

**Crate:** `bhc-gpu`
**Location:** `kernel.rs`, `context.rs`, `runtime/cuda.rs`

Tasks:
- [x] `GpuKernel` compiled kernel representation
- [x] `LaunchConfig` for grid/block dimensions
- [x] Launch parameter setup
- [x] Full kernel execution pipeline (`GpuContext::launch()` → `runtime::cuda::launch_kernel()`)
- [ ] Test: Kernel executes on GPU (requires CUDA hardware)

### 6.6 Runtime Support ✅

**Crate:** `bhc-gpu`
**Location:** `runtime/cuda.rs`, `runtime/rocm.rs`

Tasks:
- [x] CUDA runtime integration (cuBLAS)
- [x] ROCm/HIP runtime support
- [x] Device enumeration and selection
- [x] Stream and context management

### Phase 6 Exit Criteria

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
main = do
  let a = fromList [1..1000000]
  let b = fromList [1..1000000]
  print $ sum $ zipWith (+) a b
```

**Blockers:** Requires CUDA/ROCm hardware for end-to-end testing.

**Remaining effort:** ~2-3 days (end-to-end testing with GPU hardware)

---

## Phase 7: Advanced Profiles 🟡 90% COMPLETE

### 7.1 Realtime (Bounded GC) ✅

**Crate:** `bhc-rts-gc`
**Location:** `incremental.rs` (730 lines), `lib.rs` (1,500+ lines)

Tasks:
- [x] Tri-color marking infrastructure (White/Gray/Black)
- [x] `MarkState` enum (Idle/RootScanning/Marking/Remark/Complete)
- [x] SATB write barrier buffer
- [x] Pause budget configuration (`IncrementalConfig`, 500μs default)
- [x] Wire mark loop into main GC (`start_incremental_collect()`, `do_incremental_work()`, `finish_incremental_collect()`)
- [x] Pause time measurement (via `PauseMeasurement`, `PauseStats`)
- [x] Test: 32 tests pass including incremental GC cycle tests

### 7.2 Generational GC ✅

**Crate:** `bhc-rts-gc`
**Location:** `lib.rs` (1,526 lines)

Tasks:
- [x] Three-generation model (Nursery/Survivor/Old)
- [x] Write barriers for cross-generation references
- [x] Promotion logic
- [x] Collection statistics (`GcStats`)

### 7.3 Embedded (No GC) ✅

**Crate:** `bhc-rts-alloc`, `bhc-core`, `bhc-session`, `bhc-driver`
**Location:** `static_alloc.rs` (200+ lines), `escape.rs` (585 lines), `lib.rs`

Tasks:
- [x] Static allocator with fixed-size buffer
- [x] Bump pointer allocation (O(1))
- [x] No-GC design for embedded
- [x] Escape analysis (`analyze_escape()`, `check_embedded_safe()`)
- [x] EscapeStatus enum (NoEscape/EscapeReturn/EscapeCapture/EscapeStore/EscapeExternal)
- [x] Profile::Embedded with `is_gc_free()` and `requires_escape_analysis()`
- [x] CompileError::EscapeAnalysisFailed in driver
- [x] check_escape_analysis() in compilation pipeline
- [ ] Test: Bare-metal program (requires bare-metal target)

### 7.4 Arena Allocation ✅

**Crate:** `bhc-rts-arena`

Tasks:
- [x] Hot arena for ephemeral allocations
- [x] Bulk deallocation at scope end
- [x] No GC interaction

### Phase 7 Exit Criteria

**Blockers:** Requires bare-metal target hardware for testing.

**Remaining effort:** ~1-2 days (bare-metal testing with appropriate hardware)

---

## Phase 8: Ecosystem 🟡 60% COMPLETE

### 8.1 REPL 🟡

**Crate:** `tools/bhci`

Tasks:
- [x] Interactive evaluation loop (rustyline)
- [x] Full command set (`:help`, `:load`, `:reload`, `:browse`, etc.)
- [x] Profile selection support
- [ ] Expression evaluation (currently returns placeholder `Value::Int(42)`)
- [ ] Expression type inference (`:type` command — uses stale API)
- [ ] Value pretty-printing (partially implemented)
- [ ] Test: `:t map` shows type

**Status (Jan 29, 2026):** bhci compiles and launches but evaluation is stubbed. The tool was using stale imports (`bhc_core::Value`, `bhc_typeck::context::TypeContext`, `bhc_types::Type`) that were fixed to match current APIs. Real expression evaluation is not wired up.

### 8.2 Package Manager 🟡

**Crate:** `bhc-package`

Tasks:
- [x] Package description format (TOML `bhc.toml`)
- [x] Dependency resolution with semver
- [x] Build orchestration
- [x] Registry integration
- [x] Lockfile management
- [ ] Test: Integration tests have import errors (`PackageIndex` not in scope)

**Status (Jan 29, 2026):** Core code compiles but test suite has import errors (fixed). Needs further integration testing.

### 8.3 LSP Server 🟡

**Crate:** `bhc-lsp`

Tasks:
- [x] Diagnostics (errors, warnings)
- [x] Go to definition
- [x] Hover information
- [x] Completions
- [x] Document/workspace symbols
- [x] Code actions
- [ ] Independent integration testing

### 8.4 Documentation 🟡

Tasks:
- [x] User-facing documentation (`docs/getting-started.md`, `language.md`, `profiles.md`, `examples.md`)
- [ ] Developer documentation (incomplete)
- [ ] API documentation for new crates

### 8.5 IR Inspector 🟡

**Crate:** `tools/bhi`

Tasks:
- [x] IR dump formatting
- [x] Memory report display
- [ ] Integration testing

**Status (Jan 29, 2026):** bhi compiles after fixing `Allocation` missing `Clone` derive and type annotation on `current_indent`.

### Phase 8 Exit Criteria

```bash
$ bhci
bhci> :t map
map :: (a -> b) -> [a] -> [b]

$ bhc-lsp  # Starts LSP server for IDE integration
```

**Not yet verified.** REPL evaluation is stubbed.

---

## Summary Timeline

| Phase | Description | Status | Completion | Test Evidence |
|-------|-------------|--------|------------|---------------|
| 1 | Native Hello World | ✅ Complete | 100% | 6/6 E2E tests pass |
| 2 | Language Completeness | ✅ Complete | 100% | 59/59 interpreter tests pass |
| 3 | Numeric Profile | 🟡 Partial | ~55% | Tensor/Loop fusion feeds GPU/WASM + kernel report. A Core→Core list-fusion pass exists (`simplify/fuse.rs`, numeric-gated) but only `map/map` fires — `sum/map` is inert because Core erases types (`Fun(Error,Error)`), and even a firing rewrite gives no speedup because codegen boxes every `Int`. Honest native perf contract is blocked on typed Core IR + unboxed codegen, NOT a fusion pass (2026-07-03 measurement). See Phase 3 exit criteria. |
| 4 | WASM Backend | 🟡 In Progress | 75% | Valid WASM binaries; 0/6 E2E output tests pass (needs Core IR → WASM lowering) |
| 5 | Server Profile | ✅ Complete | 100% | Scheduler, STM, cancellation, deadlines all tested |
| 6 | GPU Backend | 🟡 In Progress | 80% | 2/2 mock tests pass; needs CUDA hardware |
| 7 | Advanced Profiles | 🟡 In Progress | 90% | GC, arena, embedded allocator all tested |
| 8 | Ecosystem | 🟡 In Progress | 60% | Tools compile but evaluation is stubbed |

**Overall: ~85% complete**

---

## Remaining Work

### Phase 2 (Language) - ✅ RESOLVED
All 59 interpreter tests pass:
- Fixed IO output capture: evaluator now buffers output from `putStrLn`/`print`/`putStr` operations
- Fixed nested `where` clause lowering: inner where bindings are now properly scoped and emitted as `Let` expressions

### Phase 4 (WASM) - Core IR → WASM Lowering Needed
1. ~~Debug WASM binary validation~~ — **RESOLVED**: Fixed type index encoding (log2 alignment for memops, proper type table lookup for function/import sections, Drop for _start return value)
2. WASM binaries are now **valid** and load/run in wasmtime
3. **Remaining gap**: No Core IR → WASM lowering path exists. Only Loop IR → WASM (numeric kernels) is implemented. The `_start` entry point calls a placeholder main that returns 0
4. Need to implement: Core IR expression evaluation → WASM instructions for general Haskell programs (closures, thunks, ADTs, IO actions)

### Phase 6 (GPU) - Hardware-Blocked
1. End-to-end GPU test (requires CUDA hardware)
2. PTX mock validation passes (2/2 tests)
3. AMDGCN codegen needs real hardware testing

### Phase 7 (Advanced) - Hardware-Blocked
1. Bare-metal testing for Embedded profile (requires target hardware)

### Phase 8 (Ecosystem) - Significant Work Remaining
1. **REPL (bhci):** Wire up actual expression evaluation (currently stubbed)
2. **Package Manager:** Fix test suite, integration testing
3. **LSP Server:** Independent integration testing
4. **IR Inspector (bhi):** Integration testing

---

## Immediate Next Steps

**Build Status: ✅ CLEAN** (Jan 29, 2026)
- All 33 workspace crates compile without errors
- Fixed: `bhc-rts` non-exhaustive match (added `Realtime`/`Embedded` to `bhc_get_profile`)
- Fixed: `bhc-session` missing `Realtime` profile variant
- Fixed: `bhci` stale imports (`Value`, `TypeContext`, `Type`, `lower_expr`)
- Fixed: `bhi` missing `Clone` derive on `Allocation`, type annotation on `current_indent`
- Fixed: `bhc-package` missing `PackageIndex` import in tests
- Fixed: `bhc-driver` temp directory race condition causing parallel test failures
- Removed: `bhc-prelude` stale property tests referencing non-existent modules
- Fixed: `bhc-core` evaluator IO output capture (putStrLn/print/putStr now buffer output)
- Fixed: `bhc-lower` nested `where` clause scoping (inner where bindings properly lowered)
- Fixed: `bhc-wasm` memory alignment encoding (byte alignment → log2 for WASM binary format)
- Fixed: `bhc-wasm` function/import type index encoding (proper type table lookup)
- Fixed: `bhc-wasm` _start function stack cleanup (Drop main return value)

**E2E Testing Framework: ✅ FUNCTIONAL**
- Native E2E: 6/6 pass, 1 ignored (numeric profile)
- WASM E2E: 0/6 output tests pass (binaries are valid, but placeholder main doesn't compile Haskell code)
- GPU E2E: 2/2 pass (mock mode)
- Integration tests: 59/59 pass
- Tests now support parallel execution (fixed temp dir collision)

**Priority Order:**
1. **Implement Core IR → WASM lowering** — Required for WASM backend to compile general Haskell programs
2. **Wire REPL evaluation** — bhci compiles but doesn't evaluate
3. **GPU hardware testing** — When CUDA hardware is available
4. **Interpreter IO capture** — 9 failing interpreter tests
