# BHC-BRIEF-0003 — Lazy function arguments: stop evaluating what the callee may never use

**Document ID:** BHC-BRIEF-0003
**Status:** Scoped, not started. Repro verified still failing 2026-09-03.
**Owner:** build agent
**References:** `rules/013-optimization.md` (demand analysis, worker/wrapper);
`.claude/CLAUDE.md` Phase 2.3 ("Thunks & Laziness 🟢"); `spec/BHC-BRIEF-0002`
(typed Core, which this depends on for the cheap version)
**Audited against source:** 2026-09-03

---

## Goal

BHC evaluates function arguments **eagerly**. Haskell is call-by-need. Close the
gap, or close enough of it that a program which passes a bottom it never uses
does not die.

```haskell
boom :: Int
boom = error "BOOM"
{-# NOINLINE boom #-}

myConst :: Int -> Int -> Int
myConst a _ = a

main = print (myConst 1 boom)   -- GHC: 1.   BHC: BOOM.
```

`~/Development/pandoc-harness/H7_clean.hs`, ~10s through `chain.sh link`.
Data-constructor arguments fail the same way: `fst (2, boom)`.

## Why it matters

This is the blocker under `readMarkdown`, and therefore under every document
conversion. parsec's `manyAccum` passes

```haskell
manyErr = error "combinator 'many' is applied to a parser that accepts an empty string"
```

as its empty-ok continuation. That continuation is **never called**. BHC forces
it anyway, so `many` appears broken while its control flow is provably correct.

This was **validated end to end on 2026-08-22**: eta-expanding `manyErr` in the
vendored parsec (`manyErr _ _ _ = error "…"`, semantically identical — it is only
ever the three-argument continuation) makes `many`, `many1` and the constrained
variants all return the correct answer. That is proof the eager-argument gap was
the *sole* cause of the `many` failure.

⚠️ **It did not, by itself, unblock `readMarkdown`**, which still dies at an
identical frontier (`readWithM → runPT → __closure_Text.Parsec.Prim.58 →
bhc_force(<code addr>)`). That is a **separate** bug — a raw function pointer
reaching parsec's `eerr` slot where a closure belongs. Fixing laziness is
necessary, not sufficient. Do not scope this as "and then pandoc converts".

## Mechanism

In `bhc-codegen`, `lower_direct_call_inner` and the sibling closure/apply paths
lower each argument with `self.lower_expr(arg) → value_to_ptr → push`. Arguments
are **evaluated**, never thunked.

Laziness exists for `let` (`lower_let`'s `should_thunk` + `lower_lazy`), but a
thunked variable is **forced again at every use**, including when it is merely
passed onward. Nothing in the backend distinguishes *"this position needs WHNF"*
from *"hand this to the callee untouched"*.

Some builtins ARE lazy — `const 1 undefined` returns 1, `take 3 (repeat 7)`
works — which is exactly why the gap survived this long: it is selective, so it
reads as a collection of unrelated bugs.

⚠️ `CLAUDE.md` marks "Thunks & Laziness" complete and `rules/013` describes
demand analysis as landed. Both are about **machinery existing**. Neither is
evidence that arguments are lazy. They are not.

## Two ways to do it

### A. Call-by-need properly

Thunk every non-trivial argument at every call site; force only at scrutinee and
primitive positions. Then add a demand/strictness pass so the common case does
not allocate a thunk per argument — `bhc-core` has `demand.rs` and
worker/wrapper already, currently gated to lazy profiles.

- **Correct.** Matches the language.
- **Blast radius is the whole backend.** Every call path in codegen, plus WASM,
  plus the RTS's forcing discipline.
- **Cost is dominated by the strictness pass**, not the thunking. Without it,
  every arithmetic argument allocates and every tight loop regresses.
- Needs its own gated campaign — several rounds of the 221-module sweep, the
  ladder, the battery and the GHC differential. Not an afternoon.

### B. Thunk only what can diverge

Thunk an argument only when its lowering **can fail or loop**:

- a reference to a CAF,
- an application of `error`/`undefined`/a known bottom,
- a variable already bound to a thunk.

Leave literals, variables, lambdas and saturated constructor applications eager.

- **Not call-by-need.** A non-terminating argument that is not syntactically one
  of the above still diverges.
- **But it covers the observed failures**, including `manyErr`, without a
  strictness pass and without regressing arithmetic.
- Small enough to gate in one round.
- **Risk:** it is a heuristic, and BHC's stated philosophy is predictability over
  folklore. If taken, it must be documented as a *stopgap with a named
  successor*, not as laziness.

## Recommendation

**B first, A as the real answer.** B unblocks the `many` family and buys the
ability to run more real Haskell — which is what surfaces the next bug, the way
the GHC differential surfaced three `Double` bugs in one run. A is the correct
end state and should not be attempted until Core carries types
(`BHC-BRIEF-0002`), because the strictness pass that makes A affordable needs
them.

## Prerequisites

1. **The GHC differential must be in the routine gate** before starting either.
   A change of this blast radius without ground truth is how you get a
   `float_math` — a fixture written to match the compiler rather than the
   language.
2. A **laziness fixture family** under `tier2_functions/`: unused bottom
   argument, bottom in a constructor field, bottom under `const`, bottom in an
   unforced `let`, infinite list with a finite consumer. Each must pass under
   GHC first — that is the point.

## Non-goals

- `readMarkdown` working. Separate bug, named above.
- Full call-by-need semantics under option B.
- Performance. A will regress hot loops until the strictness pass lands; that is
  expected and must be measured, not assumed.

## Definition of done

- `H7_clean.hs` prints `1` / `user-fn ok` / `2` / `tuple ok`.
- The laziness fixture family passes under the GHC differential.
- Vendored parsec's `manyErr` eta-expansion can be **reverted** and `many` still
  returns the correct answer — the honest test, since that workaround is what
  currently masks the bug in the harness.
- pandoc sweep 221/221, ladder, battery, `cargo test --all-targets
  --no-fail-fast` all green.
