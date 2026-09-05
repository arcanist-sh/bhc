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

## 1d. A codegen-builtin control name (`when`) is overwritten by an unrelated module's same-named function — the CURRENT pandoc blocker

**Symptom:** `bin_PANDOC -f native -t native doc` no longer crashes but prints
`-` then the input filename and exits 0 — the body of `when (optDumpArgs opts)`
in `convertWithOpts` (the `--dump-args` path) fires even though `optDumpArgs` is
`False`, so pandoc dumps its args and `exitSuccess` before doing the conversion.

**Root cause (minimal repro):** `App.o` references an UNDEFINED, mis-qualified
`_Text.Pandoc.Writers.OpenDocument.when`. App.hs `import Control.Monad (when)`,
but `when` resolves to `Text.Pandoc.Writers.OpenDocument.when` — an unrelated
`when :: Bool -> Doc Text -> Doc Text` (the only other top-level `when` in
pandoc). That symbol is undefined in App.o → stubbed by the harness → the
condition misreads. Two-module repro (`~/Development/pandoc-harness/repros/`):
- `ModA.hs` exports its own `when :: Bool -> Int -> Int`.
- `Main.hs`: `import Control.Monad (when)` + `import ModA (foo)` → `when` binds to
  `ModA.when` (wrong); `when False $ act` FIRES.
- `Main2.hs`: same but WITHOUT `import ModA` → `when` resolves to the builtin
  correctly. So importing an unrelated module that merely DEFINES `when` (even
  when you import only `foo` from it) corrupts the `Control.Monad.when` binding.

**What is known:** `BHC_DBG_BIND` (a temporary probe in `LowerContext::bind_value`,
reverted) shows `when` bound first to the builtin (DefId 124, kind Value) then
REBOUND to `ModA.when` (a fresh Value DefId) — the last binding wins. The final
def links to the real module's symbol (`OpenDocument.when`), which is undefined
in App.o. `Control.Monad` is a builtin module with no `.bhi`
(`register_standard_module_exports`, `lower.rs:801`), and `when` is a codegen
builtin (`lower_builtin_when`, and `builtin_info`), so it should never be
displaced by an unrelated user `when`.

**Next action:** find the rebind. `register_imported_names` (loader.rs:970) binds
an unqualified name only when `lookup_value(name).is_none() || imported_methods
.contains(name) || leaked` — so the builtin `when` (already bound) should be
safe; check whether `when` wrongly appears in ModA's `class_methods`
(`imported_methods`) or `qualified_leak_names`, or whether a second lowering pass
/ the `--package-db` preload rebinds it. The fix: a name that is a codegen
builtin control function (`when`/`unless`/`guard`/…) — or more generally, any
name explicitly imported from a builtin module — must not be displaced by a
same-named top-level function from a module the program does not import that name
from. Gate against the 221-module pandoc sweep (this touches shared import
resolution, `register_standard_module_exports`/`register_imported_names`, which
is full of collision special-cases — Djot, Citeproc, Blaze).

**Done when:** `Main.hs` prints only its non-FIRED output; `bin_PANDOC -f native
-t native /tmp/doc.native` emits the converted native AST (`[Para [Str
"Hello",Space,Str "world"]]`) instead of `-`/filename; the sweep + differential
stay green.

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
