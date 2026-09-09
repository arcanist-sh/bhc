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

**UPDATE 2026-09-05 (deeper root — the ACTUAL crash):** `BHC_DUMP_LLVM` on
`Text.Parsec.Prim` shows the `Stream` dict `runP` builds (thunk 242) has
`field_0` = the Monad Identity superclass (correct) but `field_1` (`uncons`) =
`pap_Text.Pandoc.Readers.LaTeX.Parsing.$instance_uncons_TokStream_v_Tok` —
LaTeX's `Stream TokStream m Tok` instance! parsec's generic `runP :: Stream s
Identity t => …` had its `Stream s` dictionary CONCRETIZED to an arbitrary
visible `Stream` instance (LaTeX's `TokStream`) instead of threaded as a
parameter / matched to the call's `[Char]` type. `parse … "xyz"` then calls
LaTeX's `uncons` (expecting a `TokStream`) on a `String` → the `0x1` call. db-go
exposes many `Stream` instances (`[tok]`/Prim.hs:475, ByteString, Text,
`Sources`, `TokStream`); `[tok]` is the one `[Char]` needs. This is the SAME
class as the `when`→`OpenDocument.when` collision (1d) but for INSTANCE selection
of a polymorphic constraint whose type is a variable. THIS is the real fix
target — deep dictionary-passing / instance-resolution machinery.

**SECONDARY (real, but not the crash):** `return` in the lifted continuations
resolves to `$sel_1` of the Monad dict (`>>=`), not Applicative `pure`. A broad
fix — route value-position `return` via the superclass whenever a where/let/
lambda binding (no own sig) has a Monad dict in scope
(`binding_returns_in_dict_monad` `None => true`) — was TRIED and REVERTED: it
regresses `test_tier2_user_monad`, `test_tier3_applicative_seq_transformer`,
`test_tier3_applicative_via_ap`, `test_tier3_any_all` (mis-routes `return` in
lifted bindings whose monad is NOT the enclosing dict's). A correct fix must
route only when the lifted binding's monad provably matches the in-scope Monad
dict, and make the env-captured dict reachable by `select_method_via_superclass`
(it lives in the continuation's closure env, not `dict_scope`). The IR confirms
the hop is otherwise correct: with the broad fix, `$sel_1 ($sel_0 $dMonad)` =
Applicative field 1 = `Identity.pure` (Applicative dict thunk 238 field_1).

---

**UPDATE 2026-09-09 (mechanism nailed + a working-but-too-broad fix):** the
`Stream TokStream` mis-selection is the fundep-completion in
`resolve_constrained_fn_dicts` (expr.rs ~2174). When a `Stream s Identity t`
constraint has only `m = Identity` concrete (`s`, `t` still variables), the
completion builds `pat`/`tgt` from the concrete query positions — here just
`[m]` — and EVERY `Stream` instance has `m` as a variable, so
`types_match_multi([m_var],[Identity])` matches ALL of them and `find_map` grabs
the FIRST (LaTeX's `TokStream`), inventing `s := TokStream`, `t := Tok`. That
baked LaTeX's `uncons` into runP's dict.

Adding a guard — `if !pat.iter().any(has_concrete_head) { return None }`
(`has_concrete_head` peels `App` to a `Con`/`List`/`Tuple` head) — makes runP
STOP building a concrete dict and instead THREAD its `%1` param
(`runP` becomes `tail call runPT(null, %1, …)`, verified in IR), and makes
Main's `parse` at `[Char]` resolve the real list instance. BUT it REGRESSES
`test_compile_to_executable` and `test_print_primitive`: some legit fundep
completion also has an all-variable-head matched `pat` and needs to complete. So
the guard is too broad — REVERTED. A correct guard must distinguish the
ambiguous case (MANY instances match the concrete positions — parsec's `Stream`)
from a determined one (exactly one matches). Try: complete only when the
concrete positions match a UNIQUE instance (collect all non-trivial merges,
require exactly one), rather than a head check. Gate against the two named tests
AND the parsec repros AND the sweep.

After the Stream threading is fixed, the crash MOVES (runP+208 → `parse`+96,
still a `0x1` tail-call into runPT/runParsecT) — that is the SECONDARY
`return` → `$sel_1` (Monad `>>=`) continuation issue above, which will then need
its own (targeted, not `None => true`) fix. So 1e needs BOTH.

**UPDATE 2026-09-09 (bug A FIXED — b9086a6):** the fundep-completion now
completes only when the matching instances AGREE (collect all proposals; use the
common one, else leave it polymorphic). `runP` now threads its `%1` Stream dict
(`tail call runPT(null, %1, …)`) instead of baking LaTeX's `TokStream` `uncons`;
`[Char]`/`Sources` streams still resolve. Gated green (2820 + 2 flaky-WASM that
pass 149/0 in isolation; differential 219/0). parsec STILL crashes, now purely on
bug B below.

**Bug B is the remaining blocker (confirmed with A in place):** `eok`
(`__closure_Text.Parsec.Prim.7`) resolves `return` to `field_1` of the captured
(now-correct Identity) Monad dict — `field_1 = Identity.>>=` (Monad layout
`[superclass_Applicative, >>=, >>]`). It must instead hop the superclass to the
Applicative dict and take its `pure` (`$sel_1 ($sel_0 monad_dict)`, verified
= `Identity.pure`, Applicative dict field_1). The value-position `return` arm
(expr.rs ~657) does exactly this hop but is gated by
`binding_returns_in_dict_monad`, which is false for the lambda-lifted `where`
continuation (no signature of its own). The broad relaxation
(`None => true`) was REVERTED — it regresses `test_tier2_user_monad`,
`test_tier3_applicative_seq_transformer`, `test_tier3_applicative_via_ap`,
`test_tier3_any_all` (routes `return` via the in-scope Monad dict even when the
lambda's return is for a DIFFERENT/builtin monad). A correct fix must route only
when the lifted continuation's monad matches the captured dict — needs the
ENCLOSING binding's monad (runParsecT is `Monad m =>`), which `current_binding_sig`
does not surface for a lifted binding, OR fix codegen's bare-`return`
→ Monad-`field_1` assumption to hop to Applicative. Repro: PT3.hs.

**CORRECTION 2026-09-09 (accuracy):** with fix A landed, `parse (return 7)` still
crashes at a `0x1` tail-call folded through `parse`+96 → `runPT` → `runParsecT`.
The "bug B = `return` → Monad `$sel_1`" reading above is UNCONFIRMED: the
pre-fix-A `eok` IR showed one direct `field_1` load of the captured dict, but the
post-fix-A `runParsecT` Core shows `$sel_0`/`$sel_1` (superclass) hops present —
though those cover EVERY dict access in the CPS runner (Stream `uncons`, Monad
`>>=`, Applicative `pure`), so a grep cannot attribute them to `return`. Two
targeted `return`-routing attempts both failed to change the crash:
`binding_returns_in_dict_monad` `None => true` (regressed user-monad/applicative
tests) and `|| in_scope_dict_matches` (no effect — the lifted continuation has no
recorded occurrence monad). NEXT: instruction-step `bin_PT3` under lldb through
`runPT`→`runParsecT` to the exact null `blr`/`br`, and identify WHICH dict method
(the continuation `return`, the Stream `uncons`, or a CPS continuation closure)
carries the `0x1` — do NOT assume it is `return` again. Then fix that specific
method's resolution. Repro PT3.hs / PT.hs in pandoc-harness/repros.

**CONFIRMED 2026-09-09 (instruction-stepped — bug B is real and IS `return`→`>>=`):**
stepping `bin_PT3` to the exact faulting branch shows it is
`builtin_wrapper_Identity_2e_3e_3e_3d` (`Identity.>>=`) executing `br x3` with
`x3 = 0x1`. So the continuation's `return` resolved to the Monad dictionary's
`>>=` (field 1) and, invoked, tail-branches to a garbage continuation. The
"unconfirmed" note above is superseded — bug B is the sole remaining `parse`
crash. `return` must resolve to Applicative `pure` (`$sel_1 ($sel_0 monad_dict)`),
not `$sel_1 monad_dict` (`>>=`).

The value-position `return` arm (expr.rs ~657) does the correct superclass hop
but its gate (`binding_returns_in_dict_monad`) is false for the lambda-lifted
`where` continuation. Both attempted relaxations FAILED: `None => true`
regressed user-monad/applicative tests; `|| in_scope_dict_matches` had no effect
(the lifted continuation carries no recorded occurrence monad). So `return`→`>>=`
is emitted by some OTHER path (not this arm). NEXT: instrument every
`return`-resolution site (this arm; lower_app Case 1.5 applied-`return`; any
codegen bare-`return`→Monad-field_1 fallback) with a print, recompile
`Text.Parsec.Prim`, and find which one emits `$sel_1 monad_dict` for the
continuation — then make it hop to Applicative `pure`. Tool: the lldb
step-to-crash script (`/tmp/step2.py`, limit 120000, `br set -r Prim.parse$`)
pinned the branch. Repro PT3.hs.

**✅ FIXED 2026-09-09 — the true root cause was NOT `return`→`>>=` in the
frontend (the Core hop `$sel_1 ($sel_0 $dMonad)` for `return` is CORRECT). The
crash was a codegen ARGUMENT-ORDER bug in the first-class `Identity.>>=` dict
wrapper.** Minimal pure repro (no parsec): `foo :: Monad m => m Int -> m Int;
foo m = m >>= \res -> return res` at `Identity` printed a garbage pointer;
monomorphic `Identity Int -> Identity Int` and IO both worked. In
`bhc-codegen/src/llvm/lower.rs`'s VALUE-based `lower_builtin_direct` (~47697),
`Identity.>>=` was grouped with `Identity.fmap`/`Identity.<*>`, which take the
FUNCTION first (`args[0]`). But `(Identity m) >>= k = k m` takes the monadic
VALUE first (`args[0] = m`) and the continuation second (`args[1] = k`). The
shared code treated the boxed value `m` as a closure, loaded its first word, and
branched to it (`br 0x1`). Split `Identity.>>=` into its own arm with
`func = args[1]`, `val = args[0]`. This value path is reached ONLY when the
`Monad Identity` dict is threaded through a polymorphic function (the direct
application path uses `lower_builtin_bind`, whose order was already correct — why
monomorphic worked). `parse (return 7)` → `ok: 7`, `parse (string "native")` →
`ok: native`. Gate: fmt/clippy clean, cargo test 2822/0, ghc_differential
219/0/2.

TWO ADJACENT (pre-existing, NOT this crash) gaps found while bisecting, recorded
for later — both make a polymorphic `Monad m =>` call MISS its dictionary and
shift arguments:
  (i) Maybe/list/Either have NO registered Monad/Applicative/Functor dict
      instances (only IO/Identity/ReaderT/ExceptT/StateT are in the registry);
      they are codegen builtins. `foo m = return 5` at `Maybe` → garbage.
  (ii) typeck does not PIN `m := Identity` when Identity is determined only by
       `runIdentity` in result position or by the `Identity` constructor in an
       argument (constructor/accessor types not propagated), leaving `m` a free
       var so the dict can't resolve. When `m` is pinned by a signature
       (pandoc's `Stream s Identity t`), an annotation, or the call chain, it now
       works. pandoc pins Identity explicitly, so (ii) does not block it.

**pandoc after the fix (2026-09-09):** re-swept all 221 pandoc-3.6.4 modules
with the fixed bhc — **221/221, 0 failing** (no regression from the codegen
change). The `MiniPandoc2` probe (`readMarkdown def txt >>= writeHtml5String def`
under `runIOorExplode`) now runs PAST the old `parse` crash — it prints
`INPUT_LEN` and reaches `readMarkdown` execution — then throws, because the
probe's OWN main uses stubbed external functions: `TIO.readFile`, `TIO.putStrLn`,
`T.length` (Data.Text.IO) and `def` (Data.Default) are "external package not
implemented" stubs, so `readMarkdown` is handed garbage ReaderOptions/input and
fails. `runIOorExplode` catches it, but the thrown exception's payload is a
non-null INVALID pointer, so `bhc_show_exception`'s `CStr::from_ptr` strlen-faults
(the `payload.is_null()` guard can't catch a garbage-but-non-null pointer). This
is external-stub territory, NOT the parse bug and NOT a codegen regression.
NEXT toward end-to-end conversion: give the probe real inputs — implement/vendor
`Data.Default def` for ReaderOptions and `Data.Text.IO` (readFile/putStrLn), or
build a probe that constructs ReaderOptions without `def` and feeds a `Text`
literal — so `readMarkdown` runs on real options. (Note: pandoc's INTERNAL
Data.Text usage already works — bhc-text — it's only the probe-main's direct
Data.Text.IO/Default calls that stub.)

**Data.Text.IO + Data.Default def — DONE 2026-09-09 (fae20ad, 4ba8eb7, ff9644f).**
- **Data.Text.IO / alias-qualified builtins (fae20ad):** the real bug was that
  `import qualified Data.Text as T` (and Data.Text.IO, Data.Map, ...) bound every
  alias-qualified name (`T.length`, `TIO.readFile`) to a FRESH StubValue in
  `register_standard_module_exports`, clobbering the real primops
  `define_builtins` registers under the full name. Fixed by binding the alias
  straight to the primop for a curated safe set (Map/Set/IntMap/IntSet, Data.Text,
  Data.Text.IO). Excluded Data.Text.Lazy (strict-Text result sigs) and
  Data.Sequence/Data.Foldable (only work through the by-name stub dispatch) — they
  regressed lazy_text_basic/foldable_to_list; widen only after per-primop
  verification. The probe now reads the real input file (INPUT_LEN matches).
- **Data.Default def (4ba8eb7):** `def :: Default a => a` is result-type-determined
  like `mempty`; Data.Default is external so bhc only sees pandoc's INSTANCES.
  Wired like Monoid: registered the Default class (method `def`) in hir-to-core,
  added Default to MONAD_FAMILY_CLASSES + BUILTIN_CLASS_NAMES + the is_value_class
  set (expr.rs x3), registered `def` at a FIXED DefId (10350) in both bhc-lower and
  bhc-typeck (fixed id avoids sequential-array position drift across crates),
  dropped `def` from the stub list. `def` now dispatches to Con(ReaderOptions) /
  Con(WriterOptions) — verified in the probe. NOTE: cross-module dispatch needs the
  callee's `.bhi` to preserve the concrete arg type; readMarkdown's does. A
  minimal artifact where it does NOT (`getCol :: Opts -> Int` from a stripped .bhi)
  leaves `def`'s occ type a var — same occurrence-pinning gap as the Identity case.
- **RTS (ff9644f):** bhc_show_exception only reads the payload as a C-string for
  IO/ErrorCall tags now; a raw Haskell exception (PandocError) no longer segfaults
  the top-level handler.

**NEXT for a real conversion:** `readMarkdown def txt >>= writeHtml5String def`
now runs on REAL inputs and throws a PandocError deep in execution (reported as
`<<exception>>`, exit 1). Find what it throws — likely a deeper pandoc-side stub
(many alias-qualified names outside the curated set, plus genuine externals:
Data.ByteString, Data.Time, zip) or a real parser-logic bug. Next step: make the
top-level handler show the PandocError's actual message (needs pandoc's exception
representation), or bisect readMarkdown with a trivial input.

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
