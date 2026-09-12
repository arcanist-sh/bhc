# BHC-BRIEF-0004 — Monomorphize polymorphic-monad functions at their concrete transformer stack

**Document ID:** BHC-BRIEF-0004
**Status:** The REAL pandoc writer's monad machinery runs, 2026-09-10. `pandoc-harness/
WriterProbe.hs` (`runIOorExplode (writeHtml5String def doc)` on a hand-built `Pandoc`) compiles,
links, and executes `writeHtml5String @ PandocIO` fully through the specialized stes machinery
(`bhc_eval_stes` runs) — no longer crashing in the transformer bind. It now stops at a
per-function stub (`setupTranslations`, see below), NOT a monomorphization/transformer bug.
The mechanism (`crates/bhc-core/src/monomorphize.rs`) handles same-module (`PolyW`),
cross-module (`xmod`), newtype base monad (`xmod3`), and imported newtype (`xmod4`), all
`Right 5`. Cross-module transport (bodies + newtype/synonym defs + import list, for a
transitive-closure BFS) via `.bhc` sidecars. Depends on the stes stack (DONE, 175fd86).

## Getting the real writer to run its monad machinery (2026-09-10)

Three fixes past the `xmod4` shape were needed for the actual writer:

1. **Transitive `.bhc` loading.** `PandocIO` is defined in `Text.Pandoc.Class.PandocIO`, which
   the driver imports only transitively (via `Text.Pandoc.Class`); the writer's helper chain
   spans modules too. Each sidecar now records its module's imports, and the loader BFSes the
   import closure (bounded — a whole-DB scan was O(modules²) across a sweep).
2. **Match ultimate RESULT types, not whole function types.** A cross-module occurrence type can
   carry a wrong ARGUMENT type — pandoc's `writeHtmlString'` records `Text` where its first
   parameter is `WriterState` — which made the full-type match fail. The monad variable lives in
   the result (`… -> m a` or `… -> StateT s m a`); matching only `ultimate_result` extracts it
   for both shapes and ignores the corrupt arg.
3. **Newtype/synonym unfolding transported cross-module** (already above): the driver unfolds
   `PandocIO` via the transported expansion map.

With these, `writeHtml5String` and its chain specialize (35 specializations incl. the PandocIO
instance methods), `bhc_eval_stes`/`bhc_except_t_bind_over_st` run correctly, and the writer
reaches `setupTranslations`.

### Transitive externs + constructors — LANDED (cross-module reference resolution)
A specialized clone references concrete functions and constructors from TRANSITIVELY-imported
modules (pandoc's pure `lookupMetaString`, the `Lang` constructor), but the `-c` path only
declares externs/constructor-metadata for DIRECT imports, so codegen stubbed them and they
panicked at runtime. Fix (driver, `transitive_extern_symbols`): the `.bhc` sidecar now also
carries the module's import list and its `constructors`; the driver BFSes the transitive import
closure and declares externs (`Module.name` + lambda-count arity) and constructor metadata for
everything reachable, deduped by name (direct imports win). Gated on the module actually having
`$$mono` specializations, and body-loading/mono is skipped unless `resolved` has a ground result
headed by a type con — so the polymorphic library modules that make up a sweep pay nothing.

### `setupTranslations` erased-occurrence fallback — RE-ENABLED 2026-09-12 (App regression gone)
The fallback (specialize a `Ty::Error`-typed callee at the enclosing `cur_monad`; `Ctx::cur_monad`,
save/restored in `get_or_specialize`) gets the writer past `setupTranslations`. It was reverted in
327e035 ONLY because it specialized a `ReaderT`-using function in `Text.Pandoc.App` at a concrete
ExceptT-over-StateT monad and codegen could not run `runReaderT` over that inner monad (App
221→220). With the whole `ReaderT`-over-`ExceptT`-over-`StateT` stack now implemented (64af425,
blocker 1 below), that gap is closed, so the fallback is re-enabled and App compiles again.
Gates with the fallback: cargo test 2828/0, ghc_differential 219/0/2 (no divergence on any general
transformer program — the fallback fires conservatively, only inside an active specialization whose
`cur_monad` mentions a transformer), pandoc sweep 221/221. **`WriterProbe` now runs PAST
`setupTranslations`** (it is specialized, no longer a link stub) into the real writer machinery and
hits a NEW crash — see next frontier.

### NEXT FRONTIER — `bhc_stes_then` null-Either crash (writer runtime, 2026-09-12)
`WriterProbe.hs` (`runIOorExplode (writeHtml5String def doc)`) now prints `WRITER_START` and
SIGSEGVs in `bhc_stes_then + 52` (EXC_BAD_ACCESS at 0x0). Disassembly: `stes_then` calls `m1(m1,
s1, s2)`, then `ldp x21,x19,[x0,#0x8]` loads the returned pair's `(Either, s2')`, and `ldr
x8,[x21]` faults because the Either field is NULL — i.e. some upstream stes action returned a
malformed `(null, s2')`. Not a regression (before the fallback, `WriterProbe` aborted earlier at
the `setupTranslations` link stub) and not in the new `ret_*` code (this is the pre-existing stes
= StateT-over-ExceptT-over-StateT machinery, reached only now that `setupTranslations` runs). The
2 remaining link stubs (`RTF.indentIncrement`, `RTF.listIncrement`) are RTF-only, never called on
the HTML path. Next: identify which stes action returns the null Either (a specialized clone vs a
hand-written op), likely by narrowing `WriterProbe`/adding a smaller writer probe.
Narrowed 2026-09-12: `WPMin.hs` (same probe, EMPTY block list `Pandoc (Meta M.empty) []`)
ALSO crashes identically — so the null Either is in the writer SETUP path. The first stes
action in `pandocToHtml` is `lift $ setupTranslations meta`, and it is `m1` in the crashing
`stes_then`.

**~~ROOT-CAUSED 2026-09-12 (a): null `PandocMonad` dictionary slots~~ — SUPERSEDED, was a static
red herring.** Static disassembly of `Translations.setTranslations` shows it read `modifyCommonState`
from the dict at `[x1,#0xa8]` and return null on a null slot — but a runtime lldb trace proves
`setTranslations` (and `setupTranslations`) are **NEVER REACHED** before the crash (breakpoints at
`Main.setTranslations$$mono` / `setupTranslations$$mono` / the polymorphic `Translations`/`Shared`
symbols never fire; `bhc_stes_then` fires first). The null dict slot is real but not on the crash
path, so the dictionary-resolution fix is NOT what unblocks the writer.

**ROOT-CAUSED 2026-09-12 (b, runtime lldb): a `lift` is mis-lowered as 2-arg `bhc_reader_t_lift`
where a 3-arg stes action is required.** Backtrace at the crash: `builtin_wrapper_evalStateT →
bhc_except_t_bind_over_st → bhc_eval_stes → bhc_stes_then`. That is `writeHtmlString'`'s
`evalStateT (pandocToHtml opts d) st`; `pandocToHtml`'s first statement `lift $ setupTranslations
meta` is `m1` of the opening `>>`. At the `blr` that calls `m1`, `x9 = bhc_reader_t_lift`
(lldb `image lookup` resolves it by name) — the **generic 2-arg ReaderT lift**. But the writer
stack is `StateT WriterState PandocIO` with `PandocIO = ExceptT PandocError (StateT CommonState
IO)` (confirmed in `Class/PandocIO.hs`): pure stes, **no ReaderT anywhere**. Called with the stes
3-arg convention `m1(m1, s1, s2)`, `bhc_reader_t_lift` (a `\(env,r) -> action` that just returns
its captured inner action, ignoring `r`) hands back the unevaluated `setupTranslations` action;
`stes_then` reads `[result+8]` as the `Either` field, gets a closure field ≈ null, and faults.
So `setupTranslations` never runs — the crash is the `lift` itself. This lives in the
fallback-created clone chain (`Main.pandocToHtml$$mono` / `writeHtmlString'$$mono` /
`writeHtml5String$$mono` all exist), so it is exposed by the fallback (f2387fa); the polymorphic
`HTML.o` `pandocToHtml` lowers `lift` through the dict (no concrete transformer op), so only the
concrete-monad clone picks an op — and picks `bhc_reader_t_lift` instead of `stes_lift`.
**Partial fix landed (649a65b): the stes stack had no dedicated bare-`lift` route.** A bare
`lift m` at `current=StateT, is_stes=true` (confirmed via `BHC_DBG_LIFT` during the WriterProbe
compile) fell through to `lower_builtin_state_t_lift` → a 2-arg `bhc_state_t_lift`, wrong arity for
the 3-arg stes protocol. Now routed through `lower_stes_inner_lifted` → `stes_lift` (the
value-position twin of the `throwError` short-circuit). Verified: ETZ/ETX/PolyW still pass; sweep
221/221; cargo 2828/0; diff 219/0/2.

**STILL crashes — a SECOND `lift`, lowered as `bhc_reader_t_lift` under `current=ReaderT`, remains.**
Even after the fix + a full re-sweep, `register read x9` at the crashing `bhc_stes_then` blr still
resolves to `bhc_reader_t_lift`. But the writer stack has NO ReaderT (`PandocIO = ExceptT (StateT
IO)`), and the WriterProbe-compile `BHC_DBG_LIFT` trace shows the writer lift as `current=StateT`,
never `ReaderT` — so this reader_t_lift is emitted while compiling a DIFFERENT module's clone (in
the swept DB), under a stack codegen reads as ReaderT-topped. The lift closures live in thunks, not
the clone's function body, so `llvm-objdump` of `Main.pandocToHtml$$mono` etc. shows no lift op
directly. **Leading hypothesis: the erased-occurrence fallback (f2387fa, a known mis-firing
heuristic per blocker 2) specialized a writer function at a monad whose type codegen's
`extract_transformer_stack_*` reads as ReaderT-topped** (e.g. via `Text.Pandoc.App`, which IS
ReaderT-based and whose writer clones — `App.pandocToHtml`/`writeHtmlString'$$mono` — are in the
binary). NEXT: add `BHC_DBG_LIFT`-style logging to `lower_builtin_reader_t_lift` (prints the ambient
stack whenever reader_t_lift is emitted), re-sweep capturing it, and find which module emits a
reader_t_lift for a writer action under a no-ReaderT semantic stack. If it is a fallback mis-fire,
either constrain the fallback's `cur_monad` or fix `extract_transformer_stack_*`'s reading of the
specialized monad type. Repro DB: `SNAP=snap-mono DB=pandoc-db-fix ./chain.sh {snapshot,deps,sweep}`
then `./chain.sh link WriterProbe.hs`; at the crash, `register read x9` in `bhc_stes_then`.

### One remaining blocker for the writer
1. **The whole `ReaderT`-over-`ExceptT`-over-`StateT` stack** (codegen) — ✅ **DONE 2026-09-12.**
   Verified with `pandoc-harness/big/RRT2.hs` (`runStateT (runExceptT (runReaderT comp 5)) 10`
   for `comp :: ReaderT Int (ExceptT String (StateT Int IO)) Int`, no monomorphization) →
   `Right v=15, st=10`. The earlier finding was correct — this is a whole stack, not a targeted
   `runReaderT` fix — so it was implemented stes-style: a uniform flat 3-arg closure
   `\(self, r, s) -> (Either e a, s')` (the `ret_*` ops in `lower.rs`, gated on
   `TransformerStack::is_reader_t_over_except_t_over_state_t`). It is a flatter cousin of `stes`:
   the outer layer is the read-only ReaderT (env threaded unchanged, never paired into the
   `Right`, so the payload is just `a`), and `lift inner` needs no repackaging (the inner
   `ExceptT`-over-`StateT` action already returns `(Either e a, s')`). Ops routed: `ask`/`return`/
   `get`/`put`/`modify`/`throwError`/`lift` and `>>=`/`>>` (`ret_bind`/`ret_then`), each
   bypassing the generic ReaderT+auto-lift path (which composed inconsistent representations).
   `runReaderT` reuses `lower_run_reader_t_over_state_t` (it calls the flat `comp` 3-arg with the
   captured env, yielding a 2-arg ExceptT-over-StateT closure that `runExceptT`/`runStateT`
   consume). `lift (lift get)` composes: the inner `lift get` lowers under `[ExceptT, StateT, IO]`
   (`bhc_except_t_lift_auto_over_st(get)` → `\(self,s)->(Right s, s)`) and `ret_lift` wraps it.
   Gates: cargo test 2828/0, ghc_differential 219/0/2, pandoc sweep 221/221. NEXT: re-enable the
   erased-occurrence fallback (blocker 2) — now that App's `runReaderT` lowers, that regression is
   gone — and confirm the writer specializes past `setupTranslations`.
2. **Sound recovery of erased occurrence types.** The fallback (specialize at `cur_monad`) is a
   heuristic that mis-fires (App). A principled signal for the monad of a `Ty::Error` occurrence
   (e.g. transporting the defining module's `resolved_expr_types`, or not erasing under `lift`)
   would replace it.
Pandoc will surface further codegen coverage gaps past these — each a bounded extension.

**Remaining for the REAL pandoc writer** (beyond this mechanism): (a) `PandocMonad m =>` is a
Monad SUPERCLASS with ~17 methods — the pass rewrites the Monad `>>=`/`>>` selectors but
PandocMonad methods (`getCommonState`, `logOutput`, …) still dispatch through the PandocMonad
dictionary, which must resolve to `PandocIO`'s (compiled) instance methods; (b) the seed only
fires when the `evalStateT` argument is a SEPARATE function (`pandocToHtml …`, as pandoc has),
not an inline `do`-block — an inline block's Core type is erased (`big/xmod4/` with an inline
block fails); (c) the monomorphic type-synonym-over-transformer case (`PolyW5`) is a distinct
CODEGEN-walker issue, not this pass.

## Newtype / synonym unfolding (2026-09-10)

`PandocIO`/`AppM` is a newtype over the stes stack, so a concrete instantiation reads as an
opaque `Con` and `mentions_transformer` is false — the seed would not fire. `unfold_newtypes`
expands newtype and (non-mtl) synonym constructors to their underlying types before every
transformer check; newtypes are representationally transparent, so substituting the underlying
stack into the specialized body is sound. The `newtype → underlying` map is built in the driver
from HIR `Item::Newtype` + `typed.type_aliases` (excluding the mtl identity synonyms
`Reader`/`State`/`Writer`/`Except`/`RWS`, which codegen recognizes by name), and is TRANSPORTED
in the `.bhc` sidecar so an imported concrete monad (`PandocIO`, defined in a module the driver
imports) is unfolded in the driver.

## Cross-module implementation (2026-09-10)

The concrete instance (`m = PandocIO`/`MyIO`) is pinned in the USING module's
`resolved_expr_types`, but the writer's Core BODY lives in the defining module and
`.bhi` carries no bodies. So:

- Each module writes its freshly-lowered (pre-simplifier) Core bindings to a `.bhc`
  sidecar next to its `.bhi` (`write_core_sidecar`, bincode of `Vec<Bind>`). A
  module loads the `.bhc` of every module it imports (`load_imported_core_bodies`),
  keyed by binder name — including non-exported helpers, so the whole writer chain
  is available and specializes transitively.
- **`Symbol` had to serialize as its STRING, not its interner id** (a `u32`
  meaningless across processes) — otherwise the transported names decode to garbage.
  Custom `Serialize`/`Deserialize` on `bhc_intern::Symbol`; `.bhi` already used
  `String` names so it is unaffected.
- The pass takes an `imported: &FxHashMap<String,(Var,Expr)>` and resolves an
  occurrence's source binding by VarId (local) OR by name (imported).
  `source_binding` VALIDATES the by-id hit's name, because an imported clone body's
  ids belong to the source module and collide with the importing module's — without
  the guard, `top.get(id)` returned an unrelated local binding (this exact bug made
  `poly` resolve to `runMyIO`).
- Transported bodies are alpha-renamed to fresh disjoint ids on load
  (`refresh_var_ids`), so codegen's VarId-keyed function/extern dispatch never
  confuses an imported body's internal var with a local one. Names/types untouched.

## Implementation notes (2026-09-10)

Landed `bhc-core/src/monomorphize.rs`, run in the driver right after
`lower_module_with_imports` (before the simplifier, so original `Var` spans still
key `resolved_expr_types`). Beyond the type-substitution design below, two further
codegen realities had to be handled, each found by dumping `PolyW`:

1. **The `Monad m =>` dictionary over-applies the Core type.** `poly x` is
   `App(App(poly, $dMonad), x)`, but `poly`'s recorded type mentions only `x`, so
   `Expr::ty()` collapses to `Ty::Error` and `evalStateT` mis-routed. Fixed with
   `monadic_action_ty` in `lower.rs`: when the action's `.ty()` is `Error`, use the
   spine head's ultimate result type (all `->` stripped), which is exactly the
   monad regardless of dictionary args.
2. **The do-block `>>=`/`>>` are dict-method selections, not builtins.** A
   `Monad m` do-block lowers sequencing as `$sel_1 $dMonad` / `$sel_2 $dMonad`; the
   transformer-stack `Monad` dictionary is a NULL placeholder, so the compiled body
   reads a null slot and bails (this is why the polymorphic path never ran — the
   original `poly` has the same null-guard). The pass rewrites `$sel_1 $dMonad` →
   builtin `>>=` and `$sel_2 $dMonad` → `>>` inside every specialized (concrete-
   monad) body, so codegen routes them by the concrete ambient stack to
   `stes_bind`/`stes_then` — exactly as a hand-written concrete do-block compiles.
   `modify`/`get`/`return` were already plain builtins.

Gate at implementation: `cargo test --all-targets` 2822/0; `ghc_differential` clean;
5 new unit tests. The pass is conservative (fires only for a module-local binding
with exactly one free tyvar, used at a ground transformer stack), so it does not
fire on ordinary `Monad m` code (e.g. an IO-instance call).

Remaining: (a) **cross-module** — pandoc's writer is imported, and the pass is
same-module only, so it does not yet reach `writeHtml5String`; (b) **type-synonym
expansion** (prerequisite below) — `big/PolyW5.hs` still crashes; (c) `pure`/`return`
via a dictionary `$sel_0` is not rewritten (only `>>=`/`>>`); pandoc may need it.
**Owner:** build agent
**References:** `.claude/CLAUDE.md` Phase 9.5 (Cross-Transformer Codegen); `rules/013-optimization.md` (§ Dictionary Specialization, O.4); `spec/BHC-BRIEF-0002` (typed Core IR — this needs complete Core types)
**Audited against source:** 2026-09-10

---

## Goal

A typeclass-polymorphic function whose body runs a monad-transformer computation
over its polymorphic monad variable `m` must produce correct code when it is
called at a **concrete** transformer stack. This is the last structural blocker
under pandoc's writers and readers, every one of which is
`PandocMonad m => … -> m a` and is run at the concrete
`ExceptT PandocError (StateT CommonState IO)`.

```haskell
poly :: Monad m => Int -> StateT Int m Int
poly x = do { modify (+ x); s <- get; return s }

writeThing :: Monad m => Int -> m Int
writeThing x = evalStateT (poly x) (0 :: Int)

type MyIO = ExceptT String (StateT Int IO)
main = do
  r <- fmap fst (runStateT (runExceptT (writeThing 5 :: MyIO Int)) 0)
  print r          -- GHC: Right 5.   BHC: SIGSEGV.
```

Repro `~/Development/pandoc-harness/big/PolyW.hs`, ~2s through `/tmp/ett.sh`
(staticlib link, no full pandoc).

## Why it matters

`writeHtml5String :: PandocMonad m => WriterOptions -> Pandoc -> m Text` builds a
`StateT WriterState m` internally and `evalStateT`s it. At the `runIO` boundary
`m = ExceptT PandocError (StateT CommonState IO)`, so the writer's real monad is
`StateT WriterState (ExceptT PandocError (StateT CommonState IO))` — exactly the
"stes" stack this repo just taught codegen to emit (175fd86). But the writer is
compiled **once, polymorphically**, and never sees that stack.

## Mechanism of the bug

Codegen chooses a transformer's runtime representation (closure arity) from the
STATIC return type, in `lower_function_def`
(`crates/bhc-codegen/src/llvm/lower.rs:46960-46961`):

```rust
let return_type = self.get_return_type_from_function_type(&var.ty);
let mut transformer_stack_from_type = self.extract_transformer_stack_from_type(return_type);
```

`extract_transformer_stack_recursive` (lower.rs:688-773) pushes `StateT` for
`StateT Int m Int`, recurses on `m`, and when `m` is `Ty::Var` falls through
`_ => {}` and pushes nothing → stack `[StateT]` → the 2-arg StateT-over-IO
closure (the representation documented at lower.rs:285). At a concrete
`m = ExceptT e (StateT s IO)` the correct representation is the 3-arg stes
closure. Nothing recompiles `poly`, so the single emitted body has the wrong
arity and the caller's `evalStateT` reads it off-by-one → segfault. Confirmed by
LLVM dump: polymorphic `poly` emits `bhc_state_t_get/modify/pure` (2-arg);
concrete `poly` emits `bhc_stes_*` (3-arg).

**Where the concrete instantiation actually lives (corrected 2026-09-10 by a Core
dump of `big/PolyW.hs`).** The brief originally assumed the concrete monad rides
on an `Expr::TyApp`. It does NOT for ordinary polymorphic function calls:

- The whole module has **zero** `Expr::TyApp` nodes. HIR→Core emits `TyApp` only
  for class methods / explicit `@ty` (expr.rs:4506-4516), not for a plain
  `writeThing 5 :: MyIO Int`.
- The Core **occurrence** `Var(writeThing).ty` stays the polymorphic monotype
  `Int -> (Var 55) Int`; likewise `evalStateT` inside the body carries a fresh
  `(Var 603)`. Top-level binder `Var.ty` is a monotype with FREE tyvars (no
  `Forall`); the quantified scheme lives separately in `typed.def_schemes` /
  `merged_schemes`.
- The concrete instantiation survives ONLY in typeck's span-keyed
  `resolved_expr_types: FxHashMap<Span, Ty>` (bhc-typeck lib.rs:105; the final-
  substitution version of `expr_types`). For PolyW the `writeThing` occurrence
  token (span 678–688) maps to `Int -> ExceptT String (StateT Int IO) Int` —
  the concrete `Int -> MyIO Int`. This map is already threaded into HIR→Core
  (`lower_module_with_imports(..., Some(&typed.resolved_expr_types), ...)`,
  driver lib.rs) and every Core `Expr::Var(v, span)` still carries that `span`,
  so `resolved_expr_types[span]` is the bridge from an occurrence to its concrete
  instantiation.

There is **no** existing pass that clones a polymorphic function with a concrete
type substituted into its body. `crates/bhc-core/src/specialize.rs` only inlines
`$sel_N dict` method selections; worker/wrapper is a strictness split. The
`Monad`-witness hook (`dictionary.rs:172-215`) fixes only `return`/`pure`
DISPATCH, not the stack REPRESENTATION. (Investigation 2026-09-10.)

## Design — a monomorphization pass seeded by `resolved_expr_types`

Run in HIR→Core (where `resolved_expr_types` is in hand) as a post-pass over the
lowered Core module, or as a Core→Core pass in `bhc-core` given the span→type map.
It produces extra concrete Core bindings that codegen then lowers with the right
stack — no codegen change to the single-lowering model (codegen's `functions` map
is already `VarId`-keyed and can hold many instances).

Worklist:

1. **Seed.** Walk all Core bodies. For each occurrence `Expr::Var(v, span)` where
   `v` is a top-level binding whose monotype has a free tyvar in a transformer
   position, look up `ty_c = resolved_expr_types[span]`. If `ty_c` is a concrete
   transformer stack (StateT/ReaderT/ExceptT/WriterT/IO spine, no free vars) that
   differs from `v.ty`, derive `subst` by structurally matching `v.ty` against
   `ty_c` (one-sided: only `v.ty`'s tyvars bind). Record `(v.id, subst)` and mark
   the occurrence for redirect.
2. **Specialize.** For each distinct `(v.id, subst)`, clone the binding's Core
   body and apply `subst` to EVERY type in it — `Var{ty}` on every occurrence,
   every `Expr::ty()`-bearing node (`Lit`, `Case`), the binder's own monotype —
   with the existing `bhc_types::Subst`. Fresh `VarId`, derived name
   (`v$$<mangled-monad>`).
3. **Redirect.** Replace each seeding occurrence `Var(v, span)` with
   `Var(v_spec, span)`.
4. **Recurse.** In the clone, an inner occurrence `Var(g, _)` that was
   `g :: … (Var 55) …` now reads (post-subst) `g :: … MyIO …` on its own `Var.ty`
   — so the sub-seed comes from the SUBSTITUTED occurrence type, NOT from
   `resolved_expr_types` (the clone has no spans in that map). Match `g`'s binder
   monotype against the substituted occurrence type to get `g`'s subst; enqueue
   `(g.id, subst_g)`. Iterate to fixpoint; memoize by `(VarId, canonical subst)`
   and cap depth. (In PolyW this is exactly `writeThing@MyIO` → `poly@MyIO`.)

## Prerequisite bug — expand type synonyms in the transformer-stack derivation

Independently confirmed 2026-09-10 and needed for step 2 to work when the
substituted monad is written through a `type` synonym (and generally):

`evalStateT poly0` with `poly0 :: StateT Int MyIO Int` (`type MyIO = ExceptT …`)
segfaults, while the identical program with the synonym spelled out inline
(`big/PolyW6.hs`) prints `R 5`. `extract_inner_monad_from_state_t` /
`get_transformer_layer_from_type` (lower.rs:20549, ~20560) match `Ty::Con` names
and never expand a synonym, so `MyIO` reads as an opaque con → `None` → the
StateT-over-IO path → wrong arity. Fix: expand aliases before matching in the
transformer-stack type-walkers. hir-to-core already has
`expand_type_aliases` (`crates/bhc-hir-to-core/src/context.rs:2716`) and the
alias map from typeck (`type_aliases`); either expand the occurrence types there
so Core carries no synonyms on transformer spines, or thread the alias map into
codegen and expand at the walkers. Repros: `big/PolyW5.hs` (synonym, crashes) vs
`big/PolyW6.hs` (inline, works).

## Known hazards

- **Incomplete substitution is a landmine.** A previous partial evalStateT-over-
  ExceptT attempt was reverted (2856292) as compile-then-crash. If step 2 leaves
  ANY type in the clone with the old `m`, codegen mis-picks a representation on
  that node and the program crashes silently. Substitution must be total; verify
  by dumping the clone's Core types.
- **Termination.** Polymorphic recursion / mutually recursive groups must not
  loop; memoize and cap.
- **Dictionaries.** The dropped `Monad m` dict must not be referenced in the
  clone; if it is (e.g. an explicit `>>=` via the dict), either keep the param or
  resolve those method selections to the concrete instance first.
- **Pipeline ordering.** Must run before codegen and after the type-carrying
  Core is stable; interacts with `specialize_dictionaries` and the second
  simplifier round — pick the order deliberately and gate.

## Definition of done

- `big/PolyW.hs` prints `Right 5` (polymorphic writeThing/poly at MyIO).
- `big/PolyW5.hs` prints `R 5` (synonym prerequisite).
- The pandoc writer probe advances past `evalStateT` in `writeHtmlString'`.
- Gate: `cargo test --all-targets --no-fail-fast` 2822/0; `ghc_differential`
  219/0/2; pandoc `bhc check` 221/221 (unaffected — typecheck-only).

## Non-goals

- Perf (the specialized bodies are ordinary; no unboxing).
- Full SPECIALIZE-pragma support or cross-module specialization — only the
  same-module concrete-instantiation case pandoc needs.
