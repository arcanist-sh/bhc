# BHC-BRIEF-0002 — Typed Core IR: populate the types Core already has room for

**Document ID:** BHC-BRIEF-0002
**Status:** Core payoff DELIVERED (2026-07-24, commit 82ba0b7) — Path A + Tasks 2/3/6 done; `sum/map → foldl'` fusion fires. Remaining: Tasks 4–5 (desugaring temps, deriving) and full leaf population, both gated on codegen consuming Core types. See the "Tasks 2 + 6 RESULT" note below.
**Owner:** build agent
**References:** `rules/007-ir-design.md` §2 ("Core IR MUST preserve types"); `ROADMAP.md` Phase 3 (fusion not met on native); `spec/BHC-REVIEW-0001` §4.1
**Audited against source:** 2026-07-23

---

## Goal

Make the Core IR actually carry the types it is designed to carry. Today the Core
IR is **structurally typed** but **populated with `Ty::Error`** at nearly every
node, so every type-directed pass downstream is blind. This brief scopes closing
that gap.

It is worth doing on its own — but it is also the **keystone that unblocks two
otherwise-stuck efforts**:

1. **The Numeric differentiator.** The guaranteed-fusion rewrite `sum (map f xs) →
   foldl' (\acc x -> acc + f x) 0 xs` needs the element/accumulator types to emit
   `0` and `+`. The pass exists (`bhc-core/src/simplify/fuse.rs`) and *documents its
   own defeat*: "by the time the simplifier runs, `f.ty()` is `Fun(Error, Error)`
   … the `Int` gate always [fails]" (fuse.rs:149,158). Only the type-agnostic
   `map/map` rewrite can fire. **Typed Core is necessary for the rewrite to fire at
   all.** (It is *not sufficient* for the perf contract — see Non-goals.)
2. **The Pandoc / GHC-compat moat.** Several deep typeck/lowering bugs hit this
   session trace to type erasure between phases. A Core IR that preserves types
   removes a class of "the information was there, then thrown away" failures and
   makes type-directed lowering (dictionary specialization, worker/wrapper,
   pattern-match compilation) able to rely on real types.

## Current reality (grounded, 2026-07-23)

The structure is already there; only population is missing:

- **Core IR holds types.** `bhc-core/src/lib.rs`: `Var { … ty: Ty }` (:103);
  `Expr::Lit(Literal, Ty, Span)`; `Expr::ty()` (:211) computes a node's type. The
  module doc even claims "Every expression carries its type explicitly" (:12).
- **typeck computes every node's type.** `bhc-typeck` produces
  `TypedModule { expr_types: FxHashMap<HirId, Ty>, def_schemes: FxHashMap<DefId,
  Scheme> }` (`lib.rs:89`). `expr_types` is a per-HIR-node map; HIR nodes carry
  `HirId` (`bhc-hir/src/lib.rs`, "every expression … has a unique HirId").
- **The bridge drops `expr_types`.** The driver lowers with only definition-level
  schemes: `lower_module_with_defs_and_constructors(&hir, defs, Some(&typed.def_schemes), …)`
  (`bhc-driver/src/lib.rs:1096-1099`). `typed.expr_types` is never passed. So
  `bhc-hir-to-core` has no per-expression types and fills placeholders.
- **236 `Ty::Error` sites** in `bhc-hir-to-core` do the placeholder-filling
  (deriving.rs 99, pattern.rs 42, expr.rs 37, context.rs 25, binding.rs 18,
  dictionary.rs 10, lib.rs 5).
- **`Expr::ty()` returns `Ty::Error`** for the compositional cases too (App/Lam,
  lib.rs:220,235), so even where sub-terms were typed, the whole isn't.

Net: the Core IR *structure* holds types, but the per-node type map at the
HIR→Core boundary is empty — see the Task 0 result below.

## The gate: Task 0 (go / no-go)

Before touching 236 sites, prove the per-node types actually exist and can be
looked up at lowering time.

### Task 0 RESULT (2026-07-23): GATE FAILED — premise corrected, plan revised

Task 1 (threading) was implemented, then Task 0 revealed the brief's premise
("the info exists, just thread it") is **wrong**. The per-node types are neither
produced nor keyed:

- **`TypedModule::expr_types` is never populated.** In `bhc-typeck`, the only
  references to `expr_types` are the field declaration, its `default()` init, and
  the read in `into_typed_module` — **there is no `.insert` anywhere**. Inference
  computes each node's type but discards it; the map ships empty.
- **HIR `Expr` carries no `HirId`.** The `hir::Expr` enum (`bhc-hir/src/lib.rs`)
  has a `span()` accessor but no id per variant. `bhc-lower` defines
  `fresh_hir_id()` (context.rs:2185) but **never calls it** — HirIds are not
  assigned to anything. So even if typeck recorded types, there is no key the
  lowering could recover.

So there is nothing to probe; the coverage would be 0%. This is exactly why Task 0
is a gate. **Revised prerequisite (the real first work) below.**

### Revised Task 0/1: produce + key per-node types

Two viable paths; **Path A recommended** (it is the honest typed Core IR and the
scaffolding half-exists):

- **Path A — assign HirIds + record types.** (1) Wire the existing-but-unused
  `fresh_hir_id()` into AST→HIR lowering and give `hir::Expr` an id (a field per
  variant, or a thin `Typed<Expr>`/side-table keyed by a stable id). (2) In
  `bhc-typeck/src/infer.rs`, at each `infer_expr`, record `expr_types[id] = ty`.
  (3) Lowering looks up via the already-added `ctx.expr_ty_opt(id)`. Touches
  bhc-hir, bhc-lower, bhc-typeck, bhc-hir-to-core — but it is the change the design
  (rules/007) already assumes exists.
- **Path C — top-down propagation at lowering, no HIR change.** `def_schemes`
  *are* populated. Propagate types top-down during lowering: the enclosing
  function's declared type gives its params' types; child expression types are
  computed locally from applied function types. Avoids the HIR/typeck changes but
  re-does propagation in the lowering and is weaker for polymorphic intermediates.
  Reasonable if Path A's HIR churn is too costly.

**Status of the original Task 1 (threading):** DONE and clean-compiling — the
`expr_types` param on `lower_module_with_defs_and_constructors`, the
`LowerContext.expr_types` field + `set_expr_types`, and the `expr_ty_opt` lookup
are all in place (the driver passes `&typed.expr_types`).

### Path A DONE (2026-07-23): produce + key per-node types (span-keyed) — gate now PASSES

Chose the **side-table-keyed-by-span** variant of Path A (avoids adding a `HirId`
to every `hir::Expr` variant — massive churn — for zero correctness loss on the
cases that matter). `Span` derives `Copy/Eq/Hash` and every `hir::Expr` already
exposes `span()`, so both typeck and lowering can compute the same key.

- **Produce:** `bhc-typeck/src/infer.rs` — `infer_expr` now wraps
  `infer_expr_compute` and records `expr_types.insert(expr.span(), ty)` for every
  node. `into_typed_module` already applies the final substitution to the values.
  Parents record after children, so the outer node wins on a shared (desugared)
  span.
- **Key:** `TypedModule::expr_types` and `hir-to-core::ExprTypeMap` re-keyed
  `HirId → Span`. `LowerContext::expr_ty_opt(span)` is the lookup.
- **Gate result:** a coverage probe (temporary, in `lower_expr`) measured **100%
  HIT** — 31/31 on `sum (map (\x->x*2) xs)` + `1 - x`, and 55/55 on a program with
  `case`/`where`/`do`/tuple/ADT construction. **0 MISS.** So real per-node types
  now reach lowering. The probe was removed after validating (env-var check per
  expr is a hot-path smell); re-add ~10 lines if needed.
- **Inert w.r.t. behavior so far:** lowering does not yet *use* the types
  (`Ty::Error` still emitted). Workspace tests green, Pandoc unchanged at 112.

**Remaining (was Tasks 2–6):** populate Core `Var.ty`/`Expr` from
`ctx.expr_ty_opt(expr.span())` (expr.rs + binding.rs first), make `Expr::ty()`
compositional, synthesize types for desugaring temps, and prove `sum/map → foldl'`
fires. The keystone — real types reaching lowering — is now in place.

## Tasks (in order)

1. **Thread `expr_types` into lowering** *(bhc-hir-to-core / bhc-driver)*
   Add an `expr_types: Option<&FxHashMap<HirId, Ty>>` parameter to
   `lower_module_with_defs_and_constructors` (mirror how `type_schemes` is stored via
   `ctx.set_type_schemes`, context.rs:154). Driver passes `Some(&typed.expr_types)`.
   *Acceptance:* builds; existing tests green; `expr_types` reachable in the lowering
   context.

2. **Populate real-expression types** *(bhc-hir-to-core: expr.rs, then binding.rs)*
   Where a Core `Expr`/`Var` is created directly from a HIR expression, replace
   `Ty::Error` with `ctx.expr_types.get(&hir_id).cloned().unwrap_or(Ty::Error)`.
   Do expr.rs (37 sites) and binding.rs (18) first — these are the ones the fusion
   pass reads.
   *Acceptance:* the coverage probe (Task 0) hits its threshold on the flagship
   program; `f.ty()` in fuse.rs is `Fun(a, b)` with real `a,b` for `map (*2)`.

3. **Fix `Expr::ty()` to be compositional** *(bhc-core/src/lib.rs)*
   `App(f, x)` result = codomain of `f.ty()`; `Lam(v, body)` = `Fun(v.ty, body.ty())`;
   `Let`/`Case` = branch type. Only fall back to `Error` when a child is genuinely
   `Error`. This makes populated leaves propagate upward.
   *Acceptance:* a unit test asserting `App(Lam(x:Int, x), 1).ty() == Int`.

4. **Synthesize types for desugaring temporaries** *(pattern.rs 42, context.rs 25,
   dictionary.rs 10)* — the vars with no HirId (wildcards, pattern binders, dict
   args, case scrutinee temps). Derive from context: a pattern binder's type is the
   corresponding field type from the constructor's scheme (`con_field_defs` already
   exists in typeck); a scrutinee temp's type is the scrutinee expr's type; a dict
   arg's type is the class's dictionary type. These need not be perfect on day one —
   prioritize the ones on the fusion / worker-wrapper path.

5. **Deriving** *(deriving.rs, 99 sites)* — lowest priority. Derived-method bodies
   (Eq/Ord/Show/…) are synthesized Core; give them best-effort types but they don't
   gate fusion. Can stay `Error` initially without blocking Tasks 1–4's payoff.

6. **Prove the fusion rewrite fires** *(bhc-core/src/simplify/fuse.rs)*
   With types populated, `sum (map f xs) → foldl'` should now fire (its `Int` gate
   sees a real accumulator type). Add a Core-level test: input `sum (map f xs)`,
   assert the output contains `foldl'`/the fused loop and no residual `map`.
   *Acceptance:* the rewrite fires on the flagship program (verify via
   `--dump-core-after-simpl` / `--kernel-report`).

### Tasks 2 + 6 RESULT (2026-07-24, commit 82ba0b7): THE REWRITE FIRES ✅

Done together, narrower than the plan above. `lower_expr` wraps
`lower_expr_inner` with `annotate_ty(ctx, span, core)`, which fills a Core
`Var`'s type from `ctx.expr_ty_opt(span)`. For `sum (map dbl xs)` with
`dbl :: Int -> Int`, `f.ty()` is now a real `Fun(Int, Int)`, the `Int` gate
passes, and the rewrite fires (verified end-to-end: flagship compiles under
`--profile=numeric` and prints the right answer; a temporary `BHC_FUSE_DBG`
probe confirmed "FIRED"). Regression guard: `tier4_fusion/sum_map_named`
fixture + `test_tier4_sum_map_named_numeric`.

**Key deviation — codegen is NOT ready for full leaf population.** The plan
(Task 2) said replace `Ty::Error` on every expr-derived `Var`/`Lit`. Doing so
broke codegen: native emitted mismatched-width LLVM (`icmp eq i32 %c, i64 58`
on a char-code compare) and wasm produced invalid binaries, because codegen
was written assuming Core carries `Ty::Error` and infers scalar integer widths
from context. So `annotate_ty` is scoped to **`Var` nodes whose type is
`Ty::Fun`** — the only shape the fusion gate reads, and one that carries no
scalar width so codegen stays inert. Task 3 (`Expr::ty()` compositional) was
already satisfied in `bhc-core` (App→codomain, Lam→`Fun(x.ty, body.ty())`).
Tasks 4–5 (desugaring temps, deriving) remain, and **full leaf population is
blocked on codegen learning to consume Core types** — do not broaden
`annotate_ty` before then. Lambda-mapped fusion (`map (\x->x*2) xs`) also
remains: operator vars unannotated + lambda type var unmonomorphized.

## Acceptance criteria (whole brief)

- `bhc-hir-to-core` threads `expr_types`; real-expression Core nodes carry real
  types (coverage probe ≥ threshold).
- `Expr::ty()` is compositional, not `Error`-by-default.
- The `sum/map → foldl'` fusion rewrite **fires** on `sum (map (*2) [1..N])`
  (previously inert).
- Workspace `cargo test --all-features` stays green; Pandoc `bhc check` score does
  not regress (ideally a few modules improve as type-directed lowering gets real
  types).

## Non-goals / explicitly out of scope

- **Numeric *performance* contract.** Making the fusion rewrite *fire* is this
  brief. Making it *fast* additionally needs **unboxed numeric codegen** — the
  flagship stays ~6–11 s because every `Int` is boxed even after a rewrite
  (ROADMAP Phase 3). That is a separate, subsequent effort; do not claim a numeric
  speedup from this brief without an isolated with-fusion vs without-fusion
  measurement.
- **Full type checking in Core.** We populate types from typeck; we do not add a
  Core type-checker/verifier (a nice-to-have `verify(IR)` per rules/007, later).
- **Polymorphism/System-F rigor.** `Expr::TyApp`/`TyLam` exist but this brief does
  not require making them load-bearing; monomorphic-enough types for the rewrites
  are the bar.

## Risks

- **Sparse/miskeyed `expr_types`** — the gate (Task 0) exists to catch this before
  the 236-site grind. If HIR ids don't survive lowering, that keying is the real
  first task.
- **Regressions from now-non-`Error` types** — some code may currently *rely* on
  `Ty::Error` acting as a permissive wildcard (unifies with anything). Populating
  real types could surface latent mismatches. Mitigate: land Tasks 1–2 behind the
  coverage probe, run the full suite + Pandoc score after each task, and keep the
  `unwrap_or(Ty::Error)` fallback so anything unmapped degrades to today's behavior.
- **Scope creep into unboxing** — resist. This brief ends at "the rewrite fires."
