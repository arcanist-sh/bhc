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

**Update 2026-09-04:** the env LAYOUT is consistent (fill and read agree slot 3
= k), so parserBind genuinely receives a `Text` as `k` — at an INLINED bind site
(parserBind has `{-# INLINE #-}`; the `parserBind` symbol is unused, so a
breakpoint on it never fires). `k` is a DIRECT empty `Text` (byte_len 0), not a
thunk (`BHC_DBG_FORCETEXT` never fires). Ruled out: cross-module `def`
(dispatches correctly), `return ""`, `option ""`, `*>`/`<*`/`>>` with `pure ""`.
A new codegen guard (committed) turns the silent `udf` into a named
`bhc_bad_action` "not a closure" error. Likely the same arity-over-count root as
1b — see there.

**Next action (superseded by 1b's arity hypothesis):** find where closure 96's environment is built (the capture list
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

## 1b. Option parsing crash — FIXED (a377e71)

**Was:** `bin_PANDOC -f native -t native doc` segfaulted in `bhc_force` at
`0x3ff800` inside `__closure_Text.Pandoc.App.CommandLineOptions.130`, the
`options` action fold, before readMarkdown.

**Root cause (was task 1b's hypothesis, then isolated):** the partial
application of a function extracted from a constructor FIELD. `getOpt'` pulls
an arity-2 `ReqArg (\arg opt -> …)` action out of its `OptDescr` and applies
ONE arg (`f arg`); `apply_closure_values`' `n == 1 && tail` shortcut emitted a
direct saturated call without consulting the callee's arity, so the arity-2
body ran with its second parameter read from a garbage register. Minimal repro
`FV7.hs` (6 lines, `getf (ReqA f _) s = f s`), full-fold repro `OPTS2b.hs`.

**Fix (a377e71, landed on main, gated):** branch on the closure's recorded
physical arity in that path — direct tail call for arity 0/1 (TCO preserved),
`bhc_pap_create_1` for arity > 1. Recompiling the vendored `System.Console.
GetOpt` + `App/CommandLineOptions` with the fixed compiler cleared the
closure-130 crash; pandoc now advances to blocker 1c below.

---

## 1c. fmap over a type-erased Maybe — FIXED (60979d3)

`map f <$> (optInputFiles opts <> mbArgs)` in `adjustOpts` crashed: the fmap
dispatch decided Maybe-vs-list from the container expression's own type, which is
`Ty::Error` for a `<>` result, so it fell to the IO default and applied `f` to
the whole `Just`, walking it as a list. Fixed by dispatching on the fmap head's
recorded RESULT type `f b` (`functor_result_is_maybe`, read from
`current_builtin_ty` — concrete `Maybe [FilePath]` even when the container's type
is erased). Pandoc option parsing then runs without crashing.

---

## 1d. `when` name-collision — FIXED (f35f5b0), and `queryTerminal` implemented (414162f)

`declare_external_symbols` registered imported symbols under their bare name, so
`OpenDocument.when` shadowed the `Control.Monad.when` builtin and pandoc's
`when (optDumpArgs opts)` fired on a False flag. Fixed by not registering a
bare-name external for a name that is a codegen builtin. Then `convertWithOpts`
needed `queryTerminal stdOutput` (both stubs) — implemented `bhc_query_terminal`
(via `std::io::IsTerminal`) plus `stdInput`/`stdOutput`/`stdError`. Pandoc now
runs option parsing, terminal detection, and reaches `parseFlavoredFormat`.

---

## 1e. parsec's CPS core (`runP`/`runParsecT`) crashes — the CURRENT pandoc blocker

**Symptom:** `bin_PANDOC -f native -t native doc` reaches
`Text.Pandoc.Format.parseFlavoredFormat` (parsing the `-f native` flavor) and
crashes calling a null/`0x1` fn-ptr inside `Text.Parsec.Prim.runP + 208`.

**Isolated to the parsec CORE (minimal repros in pandoc-harness/repros):**
- `PT3.hs`: `parse (return (7::Int)) "src" "xyz"` → crash (Bus error).
- `PT.hs`:  `parse (string "native") "src" "native"` → crash (SIGSEGV).
Even the SIMPLEST parser (`return 7`) crashes, so it is not parser-specific —
`runP`/`runPT`/`runParsecT` themselves are miscompiled. Recompiling the whole
parsec chain (Pos/Error/Prim/Char/Combinator/facade) + Format with the CURRENT
compiler does NOT fix it, so it is a live codegen bug, not a stale object.

**What is known (from the unoptimised IR, `BHC_DUMP_LLVM` on `Text.Parsec.Prim`):**
- `runP p u name s = runIdentity (runPT ...)`; runP tail-calls `runPT`, which
  calls `runParsecT` (Prim.ll:2651), the CPS runner.
- `runParsecT` does `tail call unParser(null, parser)` to get the parser's CPS
  function, then applies it to the state and the four continuations
  (`cok`/`cerr`/`eok`/`eerr`, built as `__closure_Text.Parsec.Prim.226/228/…`),
  each apply guarded by a bad-action (null/Text) check that routes to
  `bhc_bad_action`.
- The crash is a RAW null call (not `bhc_bad_action`), so it bypasses those
  checks — most likely the PARSER's own body calling a continuation
  (`return`'s `eok`) whose closure fn-ptr is null, or `unParser`/the ParsecT
  newtype yielding a non-closure. The bt collapses to `runP+208 -> 0x0`
  because the runPT/runParsecT frames are tail-call-folded.

**History:** memory `project_parsec_compile.md` — a minimal dict-PAP simulation
(`MinD.hs`) was fixed (prints 17, 2026-09-02), but REAL parsec's `runParsecT`
still crashes. This is the CPS/continuation-threading area, a known multi-session
problem.

**ROOT (traced 2026-09-05):** the crash is a TAIL `br` to a null pointer, folded
through `runP -bl-> runPT -tail-> runParsecT`. `runParsecT` (Prim.ll:2651) builds
its four continuations (`__closure_…1/4/7/10`, arities 3/1/3/1) each capturing
`%1` — the MONAD dict `m` — in their env, then applies the parser (all applies
bad-action-checked). `runP` calls `runPT(null, …)` and `runPT` calls
`runParsecT(null, field_0, parser, state)`, so the dict threaded into the
continuations is `null`/`field_0`. When the parser (`return 7` = `parserReturn`)
invokes a continuation and that continuation does `m (Reply …)` — i.e. calls the
monad's `return`/method via the captured (null) dict — it branches to a null fn
-ptr. So this is DICTIONARY THREADING through the CPS continuations for the
`Monad m` (here `Identity`) parameter of `runParsecT`: the dict is null/wrong.
The `bhc_bad_action` guards do not cover it because the null call is inside the
continuation-closure body (generated code), not one of runParsecT's own applies.

**Next action:** make `runP`/`runPT`/`runParsecT` thread a real `Monad Identity`
dict (not `null`) into the continuations — or specialize the Identity case so the
continuations' `return`/`>>=` use the Identity builtins instead of a dict method.
Check how the `Monad m` dict is (not) constructed at the `runP`→`runPT`→
`runParsecT` boundary; the `ptr null` first argument at each call is the smoking
gun. Verify with PT3.hs (`parse (return 7)` → `ok: 7`).
Gate against the parsec repros AND the full sweep/differential.

**Done when:** `PT3.hs` prints `ok: 7` and `PT.hs` prints `ok: native`;
`bin_PANDOC -f native -t native /tmp/doc.native` gets past `parseFlavoredFormat`.

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
