# BHC-BRIEF-0004 — Monomorphize polymorphic-monad functions at their concrete transformer stack

**Document ID:** BHC-BRIEF-0004
**Status:** OPEN. Blocker characterized + minimal repros landed 2026-09-10. Depends on the stes stack (DONE, 175fd86).
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

The concrete type IS present in the IR but thrown away. HIR→Core emits
`Expr::TyApp(inner, concrete_ty, span)` at instantiation sites
(`crates/bhc-hir-to-core/src/expr.rs:4516`; enum at `crates/bhc-core/src/lib.rs:148`,
whose `Expr::ty()` even instantiates the `Forall`). Codegen erases it
(lower.rs:47221-47224: `Expr::TyApp(expr, _ty, _span) => self.lower_expr(expr)`).

There is **no** existing pass that clones a polymorphic function with a concrete
type substituted into its body. `crates/bhc-core/src/specialize.rs` only inlines
`$sel_N dict` method selections on known dictionary tuples; worker/wrapper is a
strictness split. The `Monad`-witness hook (`dictionary.rs:172-215`) fixes only
`return`/`pure` DISPATCH for a polymorphic monad, not the stack REPRESENTATION.
(Investigation 2026-09-10, full transcript in session notes.)

## Design — a Core→Core monomorphization pass (Option A; the only option with a hook)

New pass in `bhc-core` (call it `monomorphize.rs`), run in the driver pipeline
(`crates/bhc-driver/src/lib.rs`, after `simplify`, ideally before
`specialize_dictionaries`), producing extra concrete Core bindings that codegen
then lowers with the right stack — no codegen change to the single-lowering model
(codegen's `functions` map is already `VarId`-keyed and can hold many instances).

Worklist:

1. **Seed.** Scan Core for saturated uses of a top-level binding `f` whose scheme
   quantifies a monad variable `m`, where the instantiation of `m` at the use
   site (read from the enclosing `Expr::TyApp`'s concrete `Ty`, item above) is a
   concrete transformer stack (StateT/ReaderT/ExceptT/WriterT/IO spine, no free
   vars). Record `(f, subst = {m := concrete})`.
2. **Specialize.** For each distinct `(f, subst)`, clone `f`'s Core body and apply
   `subst` to EVERY type in it — `Var{ty}`, every `Expr::ty()`-bearing node, and
   the binder's own scheme — using the existing `Subst`. Give the clone a fresh
   `VarId` and a derived name (`f$$<mangled-monad>`). Drop the now-satisfied
   `Monad m` dictionary parameter (or leave it dead for the simplifier).
3. **Redirect.** Replace the seeding call's `TyApp(Var(f), concrete)` head with
   `Var(f_spec)`.
4. **Recurse.** The clone's body may itself apply other polymorphic functions at
   `m` (e.g. `writeThing` → `poly`); seed those too. Iterate to fixpoint; memoize
   by `(VarId, canonicalized subst)` and cap depth.

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
