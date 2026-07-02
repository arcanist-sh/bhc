# BHC-REVIEW-0001 — Architecture & Ideas Assessment

**Document ID:** BHC-REVIEW-0001
**Status:** Working document (for follow-up triage)
**Date:** 2026-07-02
**Method:** Three parallel code-review sweeps (frontend/typeck, middle-end/backend, RTS/backends/stdlib) over ~265k lines of Rust + ~43k lines of Haskell stdlib, plus a live end-to-end verification, plus a read of README/ROADMAP/CLAUDE.md/rules and BHC-RFC-0001 (the Commons).
**Purpose:** Self-contained record of findings so the maintainer can work through them with any model/reviewer without needing the original session context.

---

## 0. Verdict (TL;DR)

**The ideas are right and the architecture is genuinely sound.** BHC is a real compiler — the frontend, Core IR optimizer, and native pipeline are principled implementations of the correct algorithms, and the compiler was verified live during this review (see §1). The gap is not in the ideas or the skeleton; it is that:

1. The two headline differentiators — **runtime profiles** and **structured concurrency** — are ahead of their wiring (RTS-ready, not codegen-wired).
2. **There is no live garbage collector for compiled code** (audit complete, §5.1): `bhc_alloc` is a raw malloc, the collector is a stub, and no rooting mechanism is emitted. Safe today (nothing is freed, so nothing dangles) but programs leak — fine for short-lived batch (Pandoc), fatal for long-running targets. *This was originally listed as a "potential corruption landmine"; the audit found the opposite — see §5.1.*
3. **Documentation has drifted badly behind the code**, in both directions (§5.5) — including an overstated "Generational GC ✅ Complete" (§5.1).
4. The strategic risk is **breadth vs. the single hard thing** (GHC compatibility at Hackage scale) (§7).

> **Update (2026-07-02):** The §7 step-1 GC-root audit is **complete** (§5.1). Direction decided: **defer building a real GC; keep the leak-allocator and stay Pandoc-first** (a short-lived batch where leaking is acceptable). Real GC is deferred until a long-running target (server/REPL) is on the critical path. This document has been reconciled with the audit's findings.

---

## 1. Live verification performed during this review

The status docs (ROADMAP.md, .claude/CLAUDE.md) claim the WASM backend produces invalid binaries that fail wasmtime validation. **This is stale.** Verified 2026-07-02 with `target/debug/bhc`:

```haskell
main :: IO ()
main = do
  putStrLn "Hello from BHC"
  print (sum (map (*2) [1..10 :: Int]))
```

- Native: compiles, links, runs, prints `Hello from BHC` / `110`. ✅
- `--target=wasm32-wasi`: compiles; `wasmtime` runs it, identical output. ✅

Note this program exercises the guaranteed-fusion pattern (`sum (map f xs)`) and produces correct results on both backends. The differential harness lives at `crates/bhc-e2e-tests/differential.py` (not `tests/` as some docs imply).

**Implication:** any review that trusts ROADMAP.md's status tables will reach wrong conclusions. Two of the three review sweeps in this assessment were initially misled by these docs before code inspection corrected them.

---

## 2. Assessment of the ideas

### 2.1 Compatibility-first, innovation opt-in — RIGHT

"Standard Haskell is the baseline; new behavior is opt-in and namespaced (`BHC.*`, `BHC2026`)" is the correct posture for an alternative compiler. It is the lesson from failed GHC alternatives (they diverged) and from Clang vs GCC (it didn't). Pandoc as the north-star target operationalizes it: compatibility is earned one real package at a time, not declared.

### 2.2 Runtime profiles — the genuinely novel contribution

No other Haskell implementation offers "same language, explicit behavioral contract per module." GHC gives one runtime plus folklore. Profiles-as-contracts + kernel reports (compiler tells you what it did) is a real answer to Haskell's worst practical problem: unpredictable performance. **Irony flagged below (§5.3): this is currently the least-enforced part of the implementation.**

### 2.3 The Commons (BHC-RFC-0001) — intellectually serious, one structural tension

Strengths of the RFC itself:
- The internal audit notes are honest engagement with the code, not vision hand-waving: FxHash is explicitly called out as unsafe for a commons address; `.bhi` `module_hash` is acknowledged as a placeholder (hashes only the module *name*, `bhc-interface/src/generate.rs`); alpha-normalization of Core is correctly identified as task zero.
- `hash(meaning ⊗ contract)` identity (profile/edition/extensions folded into the address) is the right generalization of Unison's move for a compiler where the same source legitimately means different things under different profiles.
- Staged rollout where each stage is independently useful is the right shape. Honest non-goals ("checkable, not true"; syntactic-not-semantic dedup stated plainly).

Tension:
- The RFC's foundation claim — "a query-based incremental compiler is already a content-addressed cache" — leans on `bhc-query`, which is **currently the most vestigial crate in the workspace**: a real salsa-style memoization/revision/cycle-detection design (`crates/bhc-query/src/lib.rs`, ~11KB) that is **not integrated into the driver's compilation pipeline**. The Commons stage 1 ("persist bhc-query") therefore has a hidden stage 0: actually route compilation through the query system.

### 2.4 Transparency principle — right, under-delivered

"Performance is transparent — the compiler reports what happened" is the right differentiator. But there is **no benchmark suite validating that the fusion guarantees deliver the promised performance** (§5.3). For a project whose brand is measurement, the absence of measurement is the most conspicuous idea/execution gap.

---

## 3. Frontend assessment (bhc-lexer/parser/ast/lower/hir/typeck/types + support crates)

**Overall: principled, low-debt, near production-grade.**

### 3.1 Type inference
- Algorithm W (HM) with let-polymorphism, mutual recursion, signatures, error recovery (`bhc-typeck/src/lib.rs` documents this explicitly; `infer.rs` ~800 lines, `unify.rs` ~1,434 lines with occurs check, type-family reduction, alias expansion).
- **Typeclasses:** constraint accumulation during inference (`context.rs`, `emit_constraint*`), post-inference `solve_constraints()`, superclass propagation. Dictionary passing, not special-casing. Principled.
- **GADTs:** branch-local type refinement via substitution save/restore around each alternative (`infer.rs` ~226–246), existential constraints handled.
- **Higher-rank:** rank-2 via subsumption check (`ctx.subsume`) in application position instead of unification; `instantiate.rs` (~326 lines). Rank-2 only — sufficient for the compatibility target, below GHC.
- **Type families:** open/closed + associated, reduction with `Reduced | Stuck | Error` results; tensor shape families (MatMulShape, Broadcast, Transpose, Concat) in `type_families.rs` (~568 lines); a `nat_solver.rs` (~800 lines) for shape arithmetic.

### 3.2 IR structure
- Clean AST → HIR → Core separation across crates; one-way dependencies (typeck never reaches back into lowering).
- Typed indices `HirId(u32)` / `DefId(u32)` via a dedicated `bhc-index` crate — prevents ID confusion, rustc-style.
- Lowering is its own crate (`bhc-lower`, ~14k lines: context/lower/desugar/loader/resolve split).

### 3.3 Layout rule
- Full Haskell 2010 layout as a state machine in the lexer: `layout_stack`, virtual brace/semi tokens, `pending_layout_is_let`, configurable tab width. Not heuristics.

### 3.4 Diagnostics
- Structured (`bhc-diagnostics`): severity levels, primary/secondary labels, codes, notes, suggestions; terminal, JSON, JSON-lines, and LSP renderers. Typecheck has ~1,255 lines of per-error emission functions with spans everywhere and error recovery (continues after first error).

### 3.5 Debt signals
- One (1) FIXME in the entire typeck crate. 34 panics, essentially all inside `#[test]` assertions. No visible copy-paste in algorithm modules.
- Large files: `builtins.rs` ~10k lines (data-heavy, acceptable), `context.rs` ~7.9k lines (cohesive but god-object-adjacent; MEDIUM refactor candidate).
- No explicit pass structure inside typeck (single monolithic check) — fine for HM, will resist advanced analyses later. MEDIUM.

---

## 4. Middle-end & backend assessment (bhc-core, bhc-hir-to-core, tensor-ir, loop-ir, codegen, driver, query)

**Overall: the optimizer is real and substantive, not scaffolding. The main risks are a monolithic codegen file and unverified GC rooting.**

### 4.1 Core IR
- Typed, GHC-System-F-style: explicit `TyApp`, coercions (Refl, Sym, Trans, Axiom, Forall, App), source locations preserved. `Lazy { e }` marker controls thunk creation in strict profiles; thunks/blackholes delegated to RTS (`bhc_new_thunk` / `bhc_enter_thunk` / `bhc_update_thunk`). Not fully ANF-enforced, but let-heavy with simple scrutinees.

### 4.2 Optimizer reality check (all claimed passes verified to exist and be substantive)
| Pass | Location | Status |
|---|---|---|
| Simplifier (beta, case-of-known-ctor, case-of-case w/ size budget, inlining w/ occurrence analysis, dead-binding elim export-aware, constant folding, top-level alias inlining) | `bhc-core/src/simplify/` (~32KB) | ✅ real; fixpoint ≤10 iters, per-pass statistics, 11 unit tests |
| Pattern-match compilation | `bhc-hir-to-core/src/pattern.rs` (~82KB) | ✅ real; Augustsson/Sestoft column-based decision trees, ctor grouping, fallthrough |
| Demand analysis | `bhc-core/src/demand.rs` (~27KB) | ✅ real; Strict/Lazy signatures, fixpoint over recursive groups; skipped in Numeric (already strict) |
| Worker/wrapper | `bhc-core/src/worker_wrapper.rs` (~15KB) | ✅ real; forces strict args via case-wrapping (WHNF), not GHC-style unboxed workers — appropriate since BHC doesn't distinguish boxed/unboxed at LLVM level |
| Dictionary specialization | `bhc-core/src/specialize.rs` (~23KB) | ✅ real; direct method selection on known dictionaries + cleanup |
| Escape analysis | `bhc-core/src/escape.rs` (~19KB) | ✅ real; stack-vs-heap hints (Embedded profile) |

### 4.3 Tensor IR / Loop IR
- Real and **wired into the driver** (Numeric profile: Core → TensorLower → fusion context → LoopLower → vectorization analysis → parallel analysis → LLVM).
- `bhc-tensor-ir` ~159KB total; `fusion.rs` alone ~95KB. TensorMeta tracks dtype/shape/strides/layout/alias per H26-SPEC. All 4 guaranteed fusion patterns implemented, plus extended patterns (Softmax, LayerNorm, Attention). Kernel report generation exists.
- **Unvalidated:** no benchmarks demonstrating fused kernels actually run at the promised speed (see §5.3).

### 4.4 LLVM codegen
- Feature-complete for the tested language surface: literals, closures (fn-ptr + captured env via `alloc_closure`), thunks/laziness, full ADT/case dispatch, typeclass dictionaries, mutual recursion + TCO, records, transformer-stack codegen (StateT/ReaderT/ExceptT/WriterT with automatic lift insertion), FFI marshalling. 190 native E2E tests pass.
- **🔴 Monolithic:** `bhc-codegen/src/llvm/lower.rs` is ~52k lines with ~686 methods in effectively one impl block. Largest maintainability risk in the codebase; every new feature × backend pays this tax.
- **🟡 Backend duplication:** WASM lowering (`bhc-wasm`, incl. `core_lower.rs`/`lower.rs`/`wasi.rs`) parallels LLVM lowering logic with no shared abstraction. Code-rot risk as the language surface grows.

### 4.5 Incremental compilation
- `bhc-query`: salsa-like memoization, revision tracking, cycle detection — **defined but not hooked into the driver**. Vestigial today; load-bearing for the Commons RFC tomorrow. See §2.3.

---

## 5. RTS, stdlib, backends, tests — and the five priority findings

### 5.1 ✅ AUDITED (2026-07-02) — There is no live GC for compiled code; it's a leak-allocator with a stub collector

**Original hypothesis (now refuted):** "missing roots → silent heap corruption when GC runs mid-expression." The audit found the opposite. The corruption landmine is **not armed**, because collection never runs during compiled execution — but there is also **no working garbage collector at all** for native compiled code.

**Confirmed by reading the source:**

| Fact | Evidence |
|---|---|
| `bhc_alloc` is a raw `std::alloc::alloc` — no threshold, no nursery, no GC trigger | `rts/bhc-rts/src/ffi.rs:137` |
| Codegen emits `bhc_alloc` at 27 sites but **never emits `bhc_free` or `bhc_gc`** → leak-forever | `crates/bhc-codegen/src/llvm/lower.rs` (grep) |
| `major_collect`/`minor_collect` are **stubs**: take `_roots` (ignored), body is `// Placeholder: Full implementation would…`, only bump a counter, free nothing | `rts/bhc-rts-gc/src/lib.rs:865,884` |
| `force_gc()` collects with an **empty** `RootSet` and is only reachable via `bhc_gc()`, which nothing calls | `rts/bhc-rts/src/lib.rs:380` |
| **No rooting mechanism** in native codegen — no shadow stack, no statepoints, no root push/pop. `RootSet::add_stack_root` is used only in unit tests. (WASM defines `gc_root_push/pop` helpers but the native path has none.) | grep across tree |
| Objects aren't **traceable**: `ObjHeader { info_ptr }` type is defined but **no info tables / pointer maps are ever emitted**; ADTs are `{tag, fields…}` with no layout descriptor a collector could follow | `crates/bhc-codegen/src/llvm/types.rs:125`; no `info_table`/`pointer_map` matches in codegen |
| The tri-color/SATB incremental marker is real but runs over an **abstract `GcPtr` graph in tests**, disconnected from the real heap | `rts/bhc-rts-gc/src/incremental.rs` |

**Interpretation:**
- **Correctness today: safe but leaks.** Because nothing is ever freed, no pointer can dangle. This is why the 190 native E2E tests and a live hello-world pass. The project's "correctness first" rule is not currently violated by the GC.
- **The real gap is completeness:** there is no live collector. Acceptable for short-lived CLIs and — critically — for the current north star (`bhc check` / compiling Pandoc is a short-lived batch process that can leak freely). **Unacceptable for any long-running program**: a server (ironic vs. the Server profile), a REPL, or streaming/batch workloads.
- The sophisticated `bhc-rts-gc` module is a **simulation** (data structures + isolated unit tests), not a collector wired to the compiled heap.

**Decision (2026-07-02, with maintainer):** **Defer building a real GC.** Keep the leak-allocator; stay Pandoc-first. Do not invest in a collector until a long-running target is on the critical path. If/when that changes, the ordered prerequisites are: (1) emit info-table headers / pointer maps at alloc sites so objects are traceable; (2) a rooting mechanism for native codegen (shadow stack is the pragmatic choice for an LLVM frontend without statepoints); (3) a real heap behind `bhc_alloc` with an allocation-count/byte threshold; (4) replace the `major_collect`/`minor_collect` stubs with a real mark-sweep (non-moving first — avoids pointer-update complexity); (5) a debug GC-stress mode (collect on every allocation) run against the E2E suite to verify rooting completeness. A non-moving mark-sweep is the minimal correct increment; the existing generational/incremental code becomes a later optimization.

**Optional near-term mitigation (not GC):** the arena infrastructure (`rts/bhc-rts-arena`) already exists; wiring numeric-hot-path temporaries to an arena would bound the worst allocation pressure without a collector. Deferred with the rest.

### 5.2 🔴 PRIORITY 2 — Structured concurrency is a facade (RTS-ready, not wired)
- The work-stealing scheduler (`bhc-rts-scheduler`, ~1,900 lines, crossbeam deques, cancellation <1ms, 13+ tests) and STM (TVar/TMVar/TQueue, ~30 tests) are real — **from Rust**.
- **No compiled Haskell code path reaches them.** `spawn`/`await`/`withScope`/`atomically` exist as stdlib type signatures (`stdlib` concurrency module ~900 lines of Haskell) with no codegen wiring; zero E2E fixtures exercise concurrency.
- "Phase 5: Server Profile ✅ COMPLETE" in ROADMAP/CLAUDE.md is therefore not true in the sense a user would mean it. The exit-criteria tests that pass are Rust-side RTS tests.

**Action:** either wire it (spawn → FFI entry points into the scheduler; at least one E2E fixture running concurrent Haskell) or relabel the docs to "RTS-ready, not yet wired" until it is.

### 5.3 🟡 PRIORITY 3 — Profiles are config, not yet contract
- Profile configs genuinely differ in the RTS (nursery 2MB→256KB→0; incremental marking + 500µs pause budget for Server; 1ms bound for Realtime; no-GC Embedded). This is real.
- But: codegen does not yet *enforce* Numeric-profile strict-by-default semantics (strictness insertion driven by profile), and the review found the profile configs are not clearly driven from compiled-code entry points.
- **No benchmark suite validates the fusion/performance contract.** Given the brand is "performance is transparent," this is the biggest idea/execution gap. The four guaranteed patterns are implemented and produce correct results (verified §1), but "fusion happened" and "fusion made it fast" are different claims; only the first is currently checkable.

**Action:** make ONE profile fully honest end-to-end — Numeric is the obvious choice: profile flag → strictness in codegen → fused kernel → benchmark vs GHC+vector baseline, with the kernel report as the audit trail. That single vertical slice validates the entire thesis.

### 5.4 🟡 GPU backend — complete-looking but unvalidated
`bhc-gpu` ~9k lines; PTX emission for Map/ZipWith/Reduce (with parallel reduction), dynamic CUDA loading, device memory transfer, kernel caching; AMDGCN structurally complete, zero tests. All green tests are **mock-mode** (2/2). No real-hardware validation, no automatic offload triggering from compiled code. Treat as "unproven" until a CUDA machine runs it.

### 5.5 🟡 Documentation drift (actively misleading, both directions)
- ROADMAP.md + .claude/CLAUDE.md say WASM binaries fail wasmtime validation → **false as of 2026-07-02** (§1).
- ROADMAP "Server Profile ✅ Complete" → overstates (§5.2).
- Pandoc TODO on disk was stale-low at one point (real measurement 79/221 modules passing vs an on-disk claim of 10).
- Status tables dated Jan 2026 presented as current.
- Consequence observed directly: 2 of 3 review sweeps initially reproduced the stale WASM claim as fact.

**Action:** one pass to make ROADMAP/CLAUDE.md status tables match reality, and date-stamp every status table. Under-claiming is as damaging as over-claiming for an evaluator/contributor.

### 5.6 What is verified solid in the RTS/stdlib
- **GC module (as a module, not as a live collector):** the *data structures and algorithms* for a generational (nursery/survivor/old) + incremental tri-color/SATB collector exist and are unit-tested in isolation (`bhc-rts-gc`, ~2,250 lines; part of the ~161 RTS unit tests). **But it is not wired to the compiled heap** — see §5.1. Compiled programs use a leak-allocator; the collector never runs. Do not read this row as "GC works."
- **Thunks:** tag-based (-1 unevaluated, -2 blackhole), atomic CAS forcing prevents duplicate evaluation, indirection after update. (These *are* live on the compiled path — thunk allocation just never gets reclaimed.)
- **stdlib approach:** Haskell source (~6.3k lines in H26/ incl. Prelude ~1.5k) over ~122 `foreign import ccall` intrinsics into the Rust RTS — sustainable architecture for the compatibility mission (it's how real Preludes work), with the caveat that the FFI surface is a breaking-change hazard between RTS and compiled artifacts.
- **Tests:** ~240 E2E fixtures in 5 tiers (hello → functions → IO/transformers → fusion → benchmarks); differential native↔WASM harness (`crates/bhc-e2e-tests/differential.py`) that classifies divergences. Gaps: no concurrency fixtures, no performance assertions, Pandoc not represented in fixtures.

---

## 6. Consolidated status table

| Component | Reality | Evidence |
|---|---|---|
| Frontend (lex/parse/layout/typeck) | ✅ Load-bearing, principled | §3 |
| Core IR + full optimizer pipeline | ✅ Load-bearing, all claimed passes real | §4.2 |
| Native LLVM codegen | ✅ Load-bearing (190 E2E) but monolithic 🔴 | §4.4 |
| Thunks / laziness | ✅ Load-bearing (but never reclaimed) | §5.6 |
| **Live GC for compiled code** | 🔴 **None — leak-allocator + stub collector; audit done, deferred** | §5.1 |
| WASM backend | ✅ Works today (docs stale) | §1 |
| Tensor/Loop IR + fusion | ✅ Wired, correct; perf unvalidated 🟡 | §4.3, §5.3 |
| Runtime profiles | 🟡 Real RTS configs; codegen doesn't enforce contracts | §5.3 |
| Scheduler + STM | 🟡 Real in Rust; zero Haskell wiring 🔴 | §5.2 |
| GPU backend | 🟡 Mock-validated only | §5.4 |
| bhc-query (incremental) | 🔴 Defined, disconnected from driver | §4.5 |
| REPL (bhci) | 🔴 Compiles, evaluation stubbed | ROADMAP |
| Docs/status tables | 🔴 Stale both directions | §5.5 |

---

## 7. Strategic assessment

The one push-back that isn't about code: **scope**. BHC currently carries six profiles, three-plus backends, a tensor IR, a GPU path, an LSP, a REPL, and a content-addressed commons vision. The frontend and middle-end prove the ability to build compiler infrastructure at production quality. But the moat is **GHC compatibility at Hackage scale** (Pandoc at 79/221 modules and climbing), and that grind competes with everything else for attention. The architecture is right; the risk is breadth.

**Recommended sequencing:**
1. ~~**GC-root audit** (§5.1)~~ — ✅ **DONE (2026-07-02).** Finding: no live GC, leak-allocator + stub collector (§5.1). Decision: **defer** building a real GC (safe-but-leaks is acceptable for the short-lived Pandoc batch); revisit only when a long-running target lands.
2. **Pandoc grind** — the compatibility moat; each fixed bug compounds. **← current focus.**
3. **One honest profile end-to-end** — Numeric: profile-driven strictness in codegen + fusion benchmarks vs GHC baseline (§5.3). This vertical slice validates the entire profiles thesis.
4. **Docs truth pass** (§5.5) — half a day, removes the credibility tax. *(GC + WASM claims partially corrected 2026-07-02; see §5.1/§5.5.)*
5. Defer/relabel: Server-profile concurrency wiring (or mark "RTS-ready, not wired"), **live GC** (§5.1), GPU real-hardware validation, query-system integration (do it when the Commons stage 1 starts, since it's stage 0 of that plan).
6. **Codegen refactor** (§4.4) — do it opportunistically or before the next backend, not as a big-bang.

---

## 8. Open questions to work through (for the next session/model)

1. ~~**GC rooting:**~~ **ANSWERED (§5.1):** no rooting discipline exists; no live GC. When revisited, the design choice is shadow stack (pragmatic for an LLVM frontend without statepoints) vs. `gc.statepoint`, and info-table/pointer-map emission is a prerequisite. Deferred by decision. The *remaining* open question if/when GC is built: shadow stack vs. statepoints, and non-moving mark-sweep vs. reviving the existing generational/incremental code.
2. **Numeric profile enforcement:** where should profile-driven strictness live — HIR-to-Core lowering (mark everything `Lazy`-free) or a Core pass? How does it interact with demand analysis being skipped in Numeric?
3. **Benchmark harness:** criterion-style Rust harness timing compiled binaries vs GHC-compiled equivalents? What baseline set (dot product, matmul tiers, fusion chains per rules/012)?
4. **Concurrency wiring:** what does `spawn`'s codegen look like — closure → `bhc_task_spawn(fn_ptr, env)` FFI? How do scopes map to RTS scope objects? What's the minimal E2E fixture?
5. **Codegen decomposition:** natural module seams for the 52k-line lower.rs (literals/closures/case/ADT/FFI/transformers)? Can WASM and LLVM share a pre-lowering (e.g., a common "shaped Core" or extended Loop IR) to kill the duplication?
6. **bhc-query integration:** route the driver through queries first, or persist the store first? (RFC-0001 stage 1 assumes the former is done.)
7. **Docs process:** should status tables be generated from test results (differential.py + cargo test output) rather than hand-maintained, to prevent recurrence of drift?
8. **Scope discipline:** which of {GPU, Realtime, Embedded, REPL, LSP} get explicitly parked (status: "frozen, not abandoned") to protect the Pandoc + Numeric-slice focus?

---

## Appendix A: Review provenance & calibration notes

- Three independent read-only code sweeps (frontend; middle-end/backend; RTS/backends/stdlib/tests), each returning evidence-cited findings; synthesized 2026-07-02.
- **Calibration:** the frontend sweep rated the frontend "production-grade"; hold that against the fact that Pandoc doesn't fully compile yet — "production-grade architecture, beta-grade coverage" is the more precise claim.
- **Corrections applied during synthesis:** (a) two sweeps repeated the stale "WASM broken" doc claim — refuted by live test (§1); (b) one sweep reported "11KB total RTS" — wrong (it measured single files); the RTS is thousands of lines across bhc-rts / bhc-rts-gc / scheduler / arena / alloc crates with ~161 tests.
- **Post-synthesis correction (2026-07-02):** §5.1 originally hypothesized a "GC rooting corruption landmine (unverified)." A follow-up source audit refuted this: there is no live GC on the compiled path at all (leak-allocator + stub collector), so no corruption is possible today. §5.1, §5.6, §6, §7, §8-Q1 and §0 were rewritten to match; ROADMAP.md (§1.4 + top status table) and .claude/CLAUDE.md GC claims were corrected in the same pass. Lesson: the sweeps inferred "GC exists as a rich module" ⇒ "GC runs" — a module's existence and its being *wired* are independent, and the review should have checked the `bhc_alloc` body before rating the GC row.
- Line counts are approximate (mix of `wc -l` and file sizes); treat as orders of magnitude, not exact.
