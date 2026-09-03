# BHC-BRIEF-0003 — Lazy function arguments: stop evaluating what the callee may never use

**Document ID:** BHC-BRIEF-0003
**Status:** Option B IMPLEMENTED 2026-09-03. Option A still open.
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
not allocate a thunk per argument — `bhc-core` has `demand.rs`
(`analyze_module` → per-argument `Strict`/`Lazy`) and worker/wrapper already,
currently gated to lazy profiles.

**A cannot be built on today's calling convention, and this is the finding that
matters.** Audited 2026-09-03:

- `value_to_ptr` lowers an `Int` argument with `build_int_to_ptr` — a BIT-CAST,
  not a heap box, despite its "box the integer" comment. An `Int` argument is
  the integer itself sitting in a pointer register.
- So a callee cannot force its parameters. `bhc_force` is tag-driven: it reads
  an `i64` at offset 0. Handed the integer `7`, it dereferences address 7.
- There is **no pointer-tagging scheme** in the RTS or codegen — nothing marks a
  word as immediate-versus-heap.

The consequence: *both sides of every call must agree on which arguments are
thunks*, and that cannot be arranged. A caller may be a direct call in this
module (controllable), a closure/apply path (not), or another module (not).
Demand signatures do not rescue this — they tell you what a function needs, not
what every caller did.

**So A's first step is not thunking. It is making an argument forceable**, by
one of:

1. **Pointer tagging** (what GHC does). Low bit set = immediate, clear = heap
   object; `bhc_force` passes immediates through untouched. Touches every
   `value_to_ptr`/`ptr_to_int`, every RTS entry point that receives a value, and
   the WASM backend. It does not need a GC (BHC's is a leak allocator today),
   which removes the usual hardest part.
2. **Uniformly boxing every argument.** Simpler to reason about, and much
   slower — an allocation per argument until the strictness pass claws it back.

Tagging is the better end state; boxing is the easier thing to measure against.

### The escape-analysis shortcut does not work — ATTEMPTED AND REVERTED 2026-09-03

Before paying for a convention change, an apparent shortcut: if a function never
ESCAPES — referenced only as the head of a direct application — then every call
site is visible, so caller and callee CAN agree which parameters are thunks
without changing how a value is represented. `demand.rs` supplies the
`Strict`/`Lazy` mask; the caller thunks lazy arguments and the callee marks
those parameters as `thunked_vars` so its uses force.

It was built, and it worked on the cases it was aimed at — `myConst 1 (loop 0)`
returned `1`, a DIVERGING argument that is not a syntactic bottom, which the
stopgap cannot do. Pandoc stayed at 221/221 and the ladder and battery passed.

**It is still unsound, and the reason generalises: escape analysis over
`core_module.bindings` cannot see every reference, because CODEGEN SYNTHESIZES
REFERENCES THAT CORE NEVER SHOWS IT** — derived instance methods and dictionary
slots among them. "Every call site is visible" was never true.

It failed on `tier3_io/derive_foldable`: `foldr (+) 10 Nothing2` crashed with
`misaligned pointer dereference: address ... is 0xa`. `0xa` is 10 — the initial
accumulator. In the `Nothing2` branch `z` is unused, so demand marked it Lazy
and the callee forced it, but that call arrives through dictionary dispatch for
the derived `Foldable`, not through `lower_direct_call_inner`, so no caller ever
thunked it. `tier2_functions/tco` also broke.

**Do not retry this without a whole-program view of references that includes
what codegen invents.** The patch is at `/tmp/callbyneed-slice.patch` for
reference. The guard fixture `tier2_functions/lazy_escaping_fn` stays: it pins
the invariant that a lazy-parameter function used as a VALUE must remain eager,
which is the shape that corrupts silently rather than failing loudly.

- **Blast radius is the whole backend**, plus WASM, plus the RTS.
- **Cost is dominated by the calling-convention change**, not the thunking and
  not the strictness pass.
- Needs its own gated campaign, several rounds of the 221-module sweep, the
  ladder, the battery and the GHC differential. Weeks, not an afternoon.

### B. Thunk only what can diverge — IMPLEMENTED

Thunk an argument only when its lowering **can diverge**:

- an application of `error`/`undefined`/`errorWithoutStackTrace`,
- a reference to a CAF **whose body is one of those**.

Leave everything else eager. Applied at both function call sites
(`lower_direct_call_inner`) and constructor applications
(`lower_constructor_application`) — `fst (2, error "BOOM")` needs the second.

**Two things the implementation taught that this brief originally had wrong:**

**A callee cannot simply force its parameters.** `value_to_ptr` means an `Int`
argument is a raw value bit-cast to a pointer, not a heap object; forcing it
would dereference an integer. `bhc_force` is tag-driven, and the RTS already
carries a guard for "an object whose first word happens to equal the thunk tag".
So blanket forcing is UNSOUND, not merely slow. Deferring an argument is
therefore only safe where the callee either never touches it, or would have died
evaluating it anyway — which is what confines B to expressions that can diverge.

**"A reference to a CAF" was too broad.** Every CAF, thunked, broke any callee
that uses the value: `runId compute`, with `compute` an ordinary nullary
binding, pattern-matched a thunk pointer and printed it instead of `30`. Caught
by the GHC differential in a fixture with nothing to do with laziness. The rule
is now "a CAF whose body is a bottom", recorded by `detect_bottom_cafs` from the
module's OWN bindings — an imported CAF's body is not there to inspect, so it
stays eager.

## Result of B

All four fixtures in `tier2_functions/lazy_*` match GHC, including the original
`myConst 1 boom` and `fst (2, boom)`, with no strictness pass and no change to
arithmetic.

**What B is not.** A non-terminating argument that is not syntactically a bottom
still diverges, and a bottom that IS used now crashes rather than printing its
message. It is a heuristic, and BHC's philosophy is predictability over
folklore — so it stands as a *stopgap with a named successor* (option A), not as
laziness.

## Recommendation

**B first, A as the real answer.** B unblocks the `many` family and buys the
ability to run more real Haskell — which is what surfaces the next bug, the way
the GHC differential surfaced three `Double` bugs in one run. A is the correct
end state. It needs typed Core (`BHC-BRIEF-0002`) for the strictness pass that
makes it affordable, and — established above — a forceable-argument
representation before any of that is even sound.

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
