# BHC-BRIEF-0005 — Monad transformers as an ordinary Haskell library

**Document ID:** BHC-BRIEF-0005
**Status (2026-10-16):** Approved plan, Phase 1 starting. Old path remains the default.

## Problem

`transformers`/`mtl` are not a library in BHC; they are special cases spread over every
stage of the compiler:

| Stage | What is hard-coded |
|---|---|
| bhc-lower | export tables mapping `Control.Monad.{State,Reader,Writer,Except,Trans,IO.Class}`, `Data.Functor.Identity` to builtin names; 33 names rebound to fixed DefIds 10000–10092 (`context.rs` `transformer_builtins`) |
| bhc-typeck | builtin `TyCon`s `StateT`/`ReaderT`/`ExceptT`/`WriterT`/`Identity`; per-transformer method schemes at fixed DefIds (`register_transformer_ops`); `MonadState`/`MonadReader`/`MonadError`/`MonadWriter` as builtin classes (`register_mtl_classes`); `lift`/`liftIO` without class constraints |
| bhc-hir-to-core | `is_builtin_monad_type` routes these monads to a "codegen fast path"; dictionary methods renamed by stack shape (`ExceptT_st.pure`); hardcoded DefId → method-name tables |
| bhc-core | the monomorphizer (BRIEF-0004) exists because codegen picks a transformer closure's arity from the concrete stack |
| bhc-codegen | ~10,000 lines of `llvm/lower.rs` (~17%): a guessed ambient `TransformerStack`, 8 stack-shape predicates, and per-shape implementations (StateT/IO, ExceptT/StateT, StateT/ExceptT/StateT "stes", ReaderT/ExceptT/StateT, WriterT/…) |
| bhc-wasm | its own transformer rewrite (`core_lower.rs`) |

Every new stack shape pandoc uses needs another special case, and the fallbacks between
them interact (the 2026-10-16 session spent most of its time on regressions caused by
fixes in this machinery). The DefId tables also disagree between crates (10060–10083
mean different things in lower/typeck vs hir-to-core).

No document gives a performance reason for the codegen path; it began as a shortcut
(2a1e908, 2026-02-04). Haskell implementations were written earlier
(`stdlib/bhc-transformers/hs`, efb9025/efed45f) and never wired in.

## Goal

`transformers` and `mtl` are ordinary Haskell modules compiled by BHC like any user
library: data types are newtypes, `Monad`/`MonadState`/… are classes with instances, and
code using them is lowered by ordinary dictionary passing. No compiler stage knows the
names `StateT`, `get`, `lift`, `liftIO`, `throwError`, ….

## Plan

**Phase 0 — IO as a first-class value (prerequisite, found 2026-10-16).** BHC's IO model is
"a value IS the action": an IO expression performs its effect when evaluated, and `IO.>>=`
applies the continuation to the already-computed result. An action bound to a name or
stored in a data structure therefore runs at most once:

```haskell
twice act = act >> act          -- prints once
main = do
  let hello = putStrLn "hello"
  hello; hello                  -- prints once
  let acts = [putStrLn "a", putStrLn "b"]
  sequence_ acts                -- a b
  sequence_ (reverse acts)      -- prints nothing
```

A library `StateT s IO` cannot work on that model (probe: user-defined `StT` over `IO`
returns a non-pair from `runStT prog 10`). Make `IO a` a closure that performs its effects
when run; `>>=` builds an action instead of executing; `main`, `catch`, `unsafePerformIO`
run actions at defined points. Test bed: the probe above plus the e2e suite.

*Why it happens:* a let-bound IO expression becomes a memoizing thunk (`bhc_force` caches
`()`), arguments are evaluated eagerly (BRIEF-0003), `sequence_`/`mapM_` receive results,
and the 8 IO-base transformer lifts capture already-run results. Codegen cannot rely on
types to spot `IO a` (most Core leaves are `Ty::Error`; `annotate_ty` excludes `IO`).

*Design (approved 2026-10-16): run mode / value mode.* Every IO expression is lowered in
one of two modes, decided syntactically (no types needed):

- **run** (the effect happens now): `run(a >>= k)` = `x ← run(a); run(k x)`;
  `run(prim args)` = direct RTS call (no closure cost in straight-line code);
  `run(e)` otherwise = evaluate `e` to an action closure and call it.
- **value** (let RHS, argument, list element, function result): an IO expression becomes
  a closure `\() -> run(expr)` — built, not executed.

`main`, `catch`/`try`/`bracket` (the RTS already takes `fn(env)`) run actions at defined
points.

*Sequencing:* Phase 0 and Phase 3 are built together as one opt-in mode,
`BHC_IO_MODEL=closure`, which enables closure IO **and** library transformers — the
~10k lines of transformer codegen assume eager IO and are deleted in Phase 5, so they are
not retrofitted. The default (eager) path is untouched until the new mode passes the e2e
suite. The WASM backend and the Core evaluator stay eager for now.

**Phase 1 — the library.** Adapt `stdlib/bhc-transformers/hs` to the standard module
names and API:

- `Data.Functor.Identity`
- `Control.Monad.Trans.Class` (`MonadTrans(lift)`), `Control.Monad.IO.Class` (`MonadIO(liftIO)`)
- `Control.Monad.Trans.{State,State.Strict,Reader,Except,Writer,Maybe}`
- mtl: `Control.Monad.{State,Reader,Except,Writer}.Class` with `get`/`put`/`state`,
  `ask`/`local`/`reader`, `throwError`/`catchError`, `tell`/`writer`/`listen`/`pass` as
  real class methods (multi-parameter classes with functional dependencies), the direct
  instances, and the lifted instances (`MonadState s m => MonadState s (ReaderT r m)`, …)
- re-export modules `Control.Monad.{State,Reader,Except,Writer}` (+ `.Strict` variants)

Plain Haskell 2010 + MultiParamTypeClasses + FunctionalDependencies + FlexibleInstances
+ UndecidableInstances. No CPP.

**Phase 2 — delivery behind a flag.** `BHC_TRANSFORMERS=library`:

- whole-program builds (all e2e fixtures) find the modules through stdlib-path source
  discovery, which today skips them via `builtin_module_set()`;
- separate compilation (`-c`, package DB, the pandoc harness) compiles them into the DB
  like parsec.

The old path stays the default until Phase 5.

**Phase 3 — make it work as ordinary code.** With the flag on, typeck registers none of
the transformer types/classes, hir-to-core treats them as user monads, codegen never sees
transformer builtin names. Fix whatever surfaces — expected: multi-parameter class +
fundep resolution (`MonadState s m`), lifted-instance chains, superclass hops,
higher-kinded instance heads. **Scoreboard:** the 35 transformer e2e fixtures (tier2: 15,
tier3: 20) passing in both modes.

**Phase 4 — pandoc in library mode.** Scoreboard: harness probe passes per run, compared
against the default mode.

**Phase 5 — flip the default and delete** the codegen transformer code, the
typeck/lower/hir-to-core transformer tables and the WASM rewrite. The monomorphizer
becomes an optimization, not a correctness requirement.

## Gating inventory for Phases 2–3 (surveyed 2026-10-17)

What must change under `BHC_IO_MODEL=closure` so `import Control.Monad.State` resolves
to `stdlib/bhc-transformers/hs` and its names are ordinary definitions. In order, for a
minimal slice (StateT over IO, whole-program build):

1. **Flag in a shared crate.** `io_closure_mode()` exists separately in bhc-codegen,
   bhc-typeck and bhc-hir-to-core. bhc-lower and bhc-core need it too; move it to a crate
   they all depend on.
2. **Driver.** Drop the transformer modules from `builtin_module_set()`, which drives
   both `discover_modules` and `compile_files_ordered`. Add the library to the search
   paths: nothing sets `stdlib_path` today, so only `BHC_STDLIB_PATH` or `-I` reach it.
3. **bhc-lower.** `define_builtins` binds `get`/`put`/`lift`/`liftIO`/`ask`/`runStateT`/…
   globally (`builtin_funcs`, plus `transformer_builtins` at fixed DefIds 10000–10092).
   `register_imported_names` binds an import only if the name is unbound, so the library
   loses. Skip those, and empty the `register_standard_module_exports` transformer arms,
   which are only a fallback when no source or `.bhi` is found. `lower_type` makes
   `TyCon`s by name, so the library `StateT` and the builtin one are the same `TyCon`;
   typeck must stop registering the builtin one.
4. **bhc-typeck.** Skip `register_transformer_ops` and `register_mtl_classes`. Gate:
   - the name-keyed arms in `register_primitive_ops` (`liftIO`, `get`, `put`,
     `runStateT`, `evalStateT`, `lift`) and in `register_lowered_builtins` (`get`/`put`,
     `*StateT`);
   - the `throwError`/`catchError`/`liftIO` DefId mapping;
   - the transformer heads in `is_builtin_instance_multi`.
5. **bhc-hir-to-core.**
   - `BUILTIN_CLASS_NAMES` lists `MonadIO`/`MonadTrans`/`MonadState`/…, so a library
     class would never get dictionary passing.
   - `is_builtin_monad_type` must answer only for IO.
   - Skip the transformer tables and instances in `register_builtins` and
     `transformer_method_name`, keeping IO's.
   - Gate the `MonadTrans` and `runStateT`-retype special cases in `expr.rs`.
6. **bhc-codegen.** `is_saturated_builtin` and the Var-as-value path pick a builtin *by
   name* unless the Var is a local function (by VarId) or a declared external (by name).
   Make `builtin_arity` return `None` for transformer names. That also covers
   `shadows_codegen_builtin`, which `-c` imports already use to module-qualify colliding
   names. Separately, `detect_operation_layer`, `is_transformer_runner` and the
   transformer arms of `lower_builtin`/`lower_builtin_direct` are still name-keyed.
7. **bhc-core.** The monomorphizer's `is_transformer_name`/`mentions_transformer` treat
   library newtypes named `StateT`/… as builtin; skip them in closure mode.

The library tree also holds an older `BHC/Control/Monad/*.hs` copy; delete it or keep it
off the search path.

## Risks

- Phase 3 carries most of the effort: dictionary passing for higher-kinded,
  multi-parameter classes is less exercised than the hand-coded paths.
- More closures are allocated. With today's leak-only allocator this is acceptable for
  short-lived batch programs (pandoc) but matters once a real GC is wired in.
- Multi-session effort.
