# BHC — Next Tasks

**Document ID:** BHC-TODO-NEXT
**Status:** Live backlog
**Created:** 2026-09-03
**Baseline:** `e7e6e62` (main, green: pandoc sweep 221/221, ladder 31, battery 10,
`cargo test --all-targets --no-fail-fast` 2822/0, GHC differential 219 agree / 0
diverge / 2 known failures)

The four tasks below were previously scattered across `spec/`, memory, and
`#[ignore]` comments. This file is the single actionable list; each task links to
its detailed home. Ordered by leverage.

Before starting any of them: the gate for a landed change is fmt + clippy
(`-D warnings`) + the pandoc sweep + ladder + battery + `cargo test
--all-targets --no-fail-fast` + `ghc_differential.py`. The differential also runs
as the `differential` CI job. Save a `git diff > /tmp/<name>.patch` before
reverting anything non-trivial.

---

## 1. `parserBind` continuation captures a `Text` in the `k` slot — the pandoc blocker

**Detailed home:** `spec/BHC-BRIEF-0003-lazy-arguments.md` (§ "readMarkdown's crash,
run to ground under lldb"); memory `project_pandoc_link.md` (S16 addendum).

**Symptom:** `~/Development/pandoc-harness/MiniPandoc2.hs` prints `INPUT_LEN 6`
then `Bus error` / `EXC_BAD_ACCESS (code=2)` at a `udf`.

**Root cause (lldb-verified, not a guess):** `__closure_Text.Parsec.Prim.96` =
`parserBind`'s `unParser (k a) s' …`. Its `env_elem_3` (the continuation `k`,
index 3 of a 5-slot env) holds a **`BhcText`** value, not a closure — the object
has `word0 = self+0x18`, `word1 = 0` (the BhcText header layout: `data_ptr =
header+24`, `offset = 0`). Calling `word0` as a fn-ptr jumps into data → `udf`.
It is an **environment mis-capture**: a `Text` (parser input, or a Text field of
the CPS state) is threaded into the slot that should hold `k`.

**NOT** laziness, arity, or the value-representation / calling-convention problem.
So this does **not** need pointer tagging (option A in BHC-BRIEF-0003).

**Next action:** find where closure 96's environment is built (the capture list
for `parserBind`'s continuation) and why a `Text` reaches index 3. Compare the
slot codegen *stores* `k` into against the slot closure 96 *loads*
(`env_elem_3`). Likely an off-by-one or a Text/continuation swap.

**Tools (committed at `e7e6e62`):**
- `BHC_DUMP_LLVM=<dir> bhc -c Prim.hs …` → unoptimised IR with named blocks.
- `BHC_DBG_CLOSURE=1` → maps `__closure_<mod>.<n>` to its Haskell binding.
- `BHC_DBG_PAP=1` → traces PAP create/call, with a create-backtrace.
- `lldb -b` reading the bad object's header was the decisive step.

**Done when:** `MiniPandoc2` gets past `runPT` (a document conversion is a
separate, later milestone — do not scope it into this task); the differential and
full gate stay green.

---

## 2. Native stdin read path segfaults

**Detailed home:** `KNOWN_FAILURES` in `crates/bhc-e2e-tests/ghc_differential.py`;
memory `project_ghc_differential.md`; `#[ignore]` on
`test_tier3_milestone_d_csv_parser_native` in
`crates/bhc-e2e-tests/tests/native_e2e.rs`.

**Symptom:** `tier3_io/stdin_echo` and `tier3_io/stdin_readln` segfault (rc=-11)
where GHC runs them; both have a `stdin.txt`, so input *is* being fed. The
`#[ignore]` note says this is a hang/crash "reproduces at least back to
`767df7b`" that "once ate a 6h runner".

**Next action:** find the native stdin read path (`getLine`/`getContents`/
`hGetLine` on `stdin`) in the RTS and codegen; reproduce with the two fixtures.

**Done when:** both fixtures match GHC, they come out of `KNOWN_FAILURES` (which
then *requires* removal — a known failure that starts passing fails the run), and
the `#[ignore]` on the csv-parser test is lifted.

---

## 3. WASM `double_to_str` truncates and has no scientific notation

**Detailed home:** memory `project_double_semantics.md`; `#[ignore]` on
`test_tier2_float_math_wasm` in `crates/bhc-e2e-tests/tests/wasm_e2e.rs`.

**Symptom:** WASM prints `1.414214` where Haskell prints `1.4142135623730951`,
and `0.001` where Haskell prints `1.0e-3`. The **native** formatter was corrected
against GHC (`68563a9`, `format_double` in `rts/bhc-rts/src/ffi.rs`) and is the
specification; WASM's `generate_double_to_str` (`crates/bhc-wasm/src/wasi.rs`) is
hand-emitted WASM and still uses the old six-digit fixed-point rule.

**Next action:** re-implement shortest-round-trip formatting (fixed in
`[0.1, 10^7)`, scientific outside, mantissa always with a `.`) in the emitted
WASM, matching native `format_double`.

**Done when:** `test_tier2_float_math_wasm` passes and its `#[ignore]` is lifted;
`differential.py` (native↔wasm) agrees on the float fixtures.

---

## 4. Guard-fallthrough join point is a compile budget, not a fix

**Detailed home:** commit `4437165`; the `JOIN_POINT_EQUATION_BUDGET` comment in
`crates/bhc-hir-to-core/src/pattern.rs`.

**Symptom:** the join point that routes a failed guard / refutable nested pattern
to the next equation is CLONED into every alternative, so it is exponential in
the equation count. It is currently bounded by
`JOIN_POINT_EQUATION_BUDGET = 12`: guarded equations always get it; unguarded
ones only while the function has ≤12 equations (`preprocessArgs` has 4,
`toBabel` ~120). Above the budget, `f (0:xs)` before `f (x:xs)` with no guard
loses its fallthrough (see `PPI`/`PPF` probes).

**Next action:** make `compile_equations_linear` return an **expression** (bind
the join point once, outside the case) instead of a `Vec<Alt>`, so the remaining
equations exist in exactly one place. Then remove the budget.

**Done when:** the budget is gone, `PPI`/`PPF` pass unbounded, `Writers.LaTeX.Lang`
(~120 equations) still compiles in seconds, and the full gate stays green.

---

## Deferred / not in this list (by deliberate decision)

- **Option A, proper call-by-need** (`spec/BHC-BRIEF-0003`): the 576-site
  calling-convention change (pointer tagging or uniform boxing). The escape-
  analysis shortcut was tried and reverted (unsound: codegen synthesises
  references Core never shows). Gated behind typed Core (`BHC-BRIEF-0002`).
  Task 1 above is NOT this — do not conflate them.
- **`readMarkdown` → HTML byte-identical to GHC**: the north-star conversion
  milestone. Blocked on task 1 and likely more beyond it.
