# Pandoc `bhc check` — Current State & Workplan

**Goal:** `bhc check` succeeds on Pandoc's library modules (excluding Template Haskell).

**2026-07-22 — 93 → 105 (+12). The #1-lever `try` fix landed (commit 740efc2).** `try` is overloaded
(Control.Exception.try `IO a -> IO (Either e a)` vs Text.Parsec.try `ParsecT s u m a -> …`); the
curated builtin handler (context.rs, was ~:3037) pinned it to the IO shape, poisoning every Parsec
module that uses `try` (forced the parser monad to `IO`, injected `Either`). Traced via a faithful
renamed copy of `CSV.hs` (`import Text.Pandoc.Parsing`) + in-context reduction: `escaped opts = try
(char 'x')` FAILED while `char 'x'`, `char 'x' >> char 'y'`, `mzero` all PASSED → `try` is the poison.
(A standalone `import Text.Parsec` repro passed because there `try` resolves to a permissive stub;
the IO builtin only wins via the `Text.Pandoc.Parsing` re-export path.) FIX: give the curated `try`
a permissive `a -> b` scheme — Parsec use preserves the parser type via the argument, Control.Exception
uses still recover `IO (Either …)` from the surrounding case/bind. **+12 reader/parser modules (CSV,
DokuWiki, HTML.Parsing, LaTeX.Citation, LaTeX.Math, Org.BlockStarts/DocumentTree/ExportSettings/Parsing,
TikiWiki, Vimwiki, Docx.Fields), ZERO regressions, workspace 2749/0**, regression test
`bhc-driver/tests/parsec_try_scheme.rs`. NOTE: the ops-table `try` (builtins.rs:1366) is still the IO
scheme (untouched) — the curated handler is what fires for imported `try`; revisit only if needed.

**2026-07-22 — 105 → 107 (+2). `fail` fix (commit 156884c), same bug class as `try`.** The remaining
`found IO` Parsec modules traced to the curated `fail` handler (context.rs, was ~:3026) pinning
`String -> IO a` — a do-block ending in `Prelude.fail "…"` (e.g. `romanNumeral`) forced IO. Restored
`MonadFail m => String -> m a` (matches ops-table). +2 (`Parsing.Lists`, `Readers.LaTeX.Macro`), zero
regressions, 2750/0, regression test `fail_is_monad_polymorphic`.

**`Format` localized (2026-07-23, DEEP — not a quick fix).** Faithful renamed-copy + in-context
reduction of `parseFlavoredFormat`: the `expected (t, Text) found Text` on `(prefix, spec') = case
splitExtension … of …` reproduces ONLY when `prefix` (1st tuple component) is USED inside the parser
action `fixSourcePos` while `spec'` (2nd component) is the `parse` stream. Minimal trigger: replacing
`fixSourcePos` with one that references `prefix` (e.g. `T.length prefix`) FAILS; one that doesn't
PASSES. So it's an emergent pattern-binding/inference interaction — a where-bound tuple whose two
components flow into a polymorphic parser context (one as the `Stream s`, one used at `Text`) collapses
the tuple to `Text`. Deep typeck (pattern-binding monomorphism / stream-type propagation), not a
curated-scheme win. Recorded; deprioritized.

**2026-07-23 — 107 → 108 (+1). Tuple-as-Functor unify fix (commit 3fc91d9).** bhc stores tuples as a
dedicated `Ty::Tuple` variant, so `f a` (App) couldn't unify with `(x,y)` → `fmap`/`<$>` over a tuple
failed. Added an `App(f,a)` ↔ `Ty::Tuple` bridge in the unifier (mirrors the existing App↔List bridge),
treating `(x1..xn)` as `((,..,) x1..x_{n-1}) xn`. Flips `Writers.JATS.References`
(`T.dropWhile … <$> T.break … val`). Zero regressions, 2751/0, test `tuple_functor.rs`. Individual-module
fix (+1), as expected at this stage.

**2026-07-23 — 110 → 111 (+1). `module X` re-export constructor shadowing FIXED (commit pending).** `doTerm Translations.Figure` failed: `Figure` typed as `t->t->t->Block`. The driver ALREADY handled `module X` re-exports (`build_module_exports_from_hir`, via `hir.exports` which DOES preserve `module X` as `Export{name=<module>, children=All}`), but its registry merge scanned EVERY module by constructor NAME and picked the wrong same-named ctor: `Text.Pandoc.Definition.Figure :: …->Block` (arity 3) shadowed `Translations.Types.Figure :: Term` (nullary). FIX: before the name-based merge, merge each `module X` re-export from X's OWN registry entry first (`.or_insert`), so the right ctor wins. Deep 5-layer trace (registry→lower_with_registry cache→process_imports→resolve_qualified_constructor→typeck ctor-scheme fallback) but a clean ~25-line fix. +1 (LaTeX.Inline), zero regressions, 2755/0, test `module_reexport_constructor.rs` (correctness check; strict guard is the pandoc measurement since the bug is FxHashMap-order-dependent). General correctness win — `module X` re-exports now resolve to the intended definitions.

**2026-07-23 — 109 → 110 (+1). Registered skylighting type aliases (`SyntaxMap = Map Text Syntax`, `SourceLine = [Token]`, `Token = (TokenType, Text)`) in the builtin-alias table (lib.rs) so they expand in the unifier → `Highlighting` flips. Zero regressions; test `skylighting_aliases.rs`.**

**2026-07-23 — 108 → 109 (+1). `sum`/`product` were pinned `[Int] -> Int` (builtins.rs:1895 + context.rs curated); made `forall a. [a] -> a` so `sum [2.0,3.0] :: Double` works → `Readers.HTML.Table` flips. Corrects the earlier "premature Num defaulting" note — that was a MISDIAGNOSIS (instrumenting showed `Num` was already on `Double` at solve time, no defaulting fired); the solver's defaulting order is fine.**

**Re-triage at 107:** **13 tuple/shape, 6 other, 5 No-instance, 4 Parsec-monad, 1 Arrow, 1 RWS.** The
easy "curated handler pins IO" batch is EXHAUSTED (try, fail were the two). Remaining levers are
diverse/deeper, NOT single-combinator batches: the 4 Parsec-monad now have distinct causes
(`Parsing.General`: `Parsec` vs `ParsecT Sources` — likely a `type Parsec s u = ParsecT s u Identity`
alias-expansion gap; `LaTeX.Parsing`: `[]` vs NonEmpty/IntMap; infinite-type). The 13 tuple/shape are
also diverse (`Format` splitExtension-context emergent; `JATS.References` `(t Text)` vs `(Text,Text)`;
`Chunks` Tree vs fn). No obvious next batch — pick individual near-passing modules (`HTML.Table`:
`expected Int found Double` numeric-defaulting; `Format`) or investigate the `Parsec` alias gap (could
be a small multi-module lever). Scan of curated IO-pinned handlers (context.rs) found the rest are
genuine IO fns (catch/bracket/openFile/…) — correctly pinned.

**2026-07-21/22 — TRIAGE of all 44 typecheck-failing modules (enabled by the new labeled error output,
commit c13a9dd: batch `check` now prints `Type errors in <module>:`). #1 LEVER IDENTIFIED.** Categorized
every failing module by its errors (`python3` over the labeled full-tree run). Distribution:
- **23  Parsec-monad** — `expected ((ParsecT …)/(Parsec Text) …), found IO`. **THE dominant lever by
  far.** Parser do-blocks / combinator expressions infer as `IO` instead of the `Parsec`/`ParsecT`
  monad. Blocks `Parsing.Lists`, `CSV`, and ~21 more reader/parser modules. Fixing this one inference
  bug could flip up to ~23 modules (modulo layered blockers) — worth far more than any per-name fix.
- 14  tuple/shape (e.g. `Format`: `expected (t, Text) found Text` at `case splitExtension … of (_,"")`;
  `JATS.References`: `expected (t Text) found (Text, Text)`)
- 5   other-mismatch (under-applied fns, occurs-check/infinite type in `Parsing.GridTable`)
- 2   No-instance;  1  RWS-monad;  1  Arrow (`ODT.Arrows.Utils`, Either/`***`/`>>^` stubs).

**The Parsec-monad bug is EMERGENT — it does NOT isolate in minimal repros** (same pattern as `Format`
and the old `Walk` saga): a standalone `Parser` using `try`/`char`/`noneOf`/`>>` type-checks fine, but
the full module fails. Ruled out the obvious single cause: `try` IS globally schemed as
`Control.Exception.try :: IO a -> IO (Either e a)` (builtins.rs:1366) which *looks* like the poison
(the `found IO` + `found (Either t t)` errors match it exactly), and 23/23 Parsec-fail modules use
`try` — BUT a minimal `Parser` using `try` still passes (it resolves to the Parsec `try` when
`Text.Parsec` is imported). So the IO leak is a full-module interaction, not one combinator. **NEXT
(the real #1 effort):** instrument the constraint solver on one mid-size Parsec module (e.g. `CSV`,
`escaped`/`pCSVQuotedCell`) to see where `IO` first enters the parser's monad var — likely a builtin
Parsec combinator whose scheme pins a concrete monad (IO) instead of a fresh `m`, OR `<|>`/`>>`/`mzero`
defaulting. This is deep typeck work (a focused session), not a curated-scheme quick win, but it is
the highest-payoff target on the board. Triage script + labeled output make re-running trivial.

**2026-07-21 — 92 → 93. External `QName` record construction FIXED (the 2026-07-20b OOXML lead).**
`Text.Pandoc.XML.Light.QName` is stubbed by name only, so the generic constructor fallback
(`context.rs`) gave it a bare fresh-var scheme with NO field defs. Record syntax
`QName{ qName=.., qURI=.., qPrefix=.. }` then inferred as a partial function (`t -> t`) instead of
`QName`, breaking `Text.Pandoc.Writers.OOXML`; positional `QName a b c` already worked via permissive
fresh-var unification. FIX: register QName's constructor scheme (`Text -> Maybe Text -> Maybe Text ->
QName`) AND its `con_field_defs` by name in the fallback's `match name` block. Regression:
`bhc-driver/tests/qname_record_construction.rs` (record + positional). Workspace `cargo test
--all-features` **2749/0** (needs `LIBRARY_PATH=/opt/homebrew/opt/openblas/lib` — openblas is keg-only;
a bare `--all-features` fails to link `bhc-numeric`). `Writers.OOXML` flips → **93 passed, 76 failed,
52 skipped**. Committed `cbfcbf5` (unpushed). This is the **4th per-name patch** for the builtin-list
drift class — see the measured drift map + unification workplan below.

**BUILTIN-LIST DRIFT — measured map + unification workplan (2026-07-21).** The recurring per-name
patches (`optional`, `getModificationTime`, `QName`, …) are all symptoms of ONE structural bug:
bhc-lower's `builtin_funcs` (`bhc-lower/src/context.rs:344`, whose iteration at :1098 assigns each
builtin its REAL sequential `DefId`) and bhc-typeck's `ops` vec (`bhc-typeck/src/builtins.rs:621`,
whose loop at :5842 registers scheme *i* at `DefId(next_id+i)`) are ASSUMED identical but have drifted.
Static diff of the two ordered name lists (`scratchpad/drift.py`):
- lowering `builtin_funcs` = **670** entries (source of truth for name→DefId); typeck `ops` = **343**.
- **First divergence: index 53** — after `void` (52) lowering continues with the Reader/State family
  (`liftIO, ask, asks, local, reader, get, gets, put, modify, …`) while typeck's ops continues with
  the folding family (`filterM, foldM, foldM_, replicateM, replicateM_, zipWithM, …`). Shifted from
  there on and never re-aligns.
- **290 / 343 (84%) of positional entries land on the WRONG DefId** → a builtin's real DefId carries
  some other op's scheme. This is the collision class the 07-20c re-alignment repairs.
- 325 lowering names have no ops entry (permissive fresh-var path); 1 ops-only name (`getMaskingState`).

Why the positional pass is vestigial-or-harmful: `register_value` records every op *by name* too, and
`register_lowered_builtins` (`context.rs:1120`) applies those name-keyed schemes at the REAL lowering
DefIds. So by-name registration is what actually types builtins; the positional `DefId(next_id+offset)`
pass mostly just plants wrong schemes that the collision-only guard (`context.rs:1958`) then repairs.

**The fix, two independent parts (score is the oracle — build `bhc`, re-check Pandoc each step):**
- **Part 1 (structural):** stop assigning ops schemes at positional DefIds; register them ONLY by name
  and let `register_lowered_builtins` apply at lowering's real DefId. Then no wrong scheme can land on
  a real DefId and the collision-only guard becomes unnecessary. Net simplification.
- **Part 2 (semantic — the trap):** making ops schemes authoritative-by-name EXPOSES every wrong ops
  scheme previously masked by landing on the wrong DefId. The 07-20c experiment already hit this
  (name-key every builtin → 89→76, 14 regressions): some names work ONLY by falling through to the
  permissive path. So Part 1 must be gated by auditing **ops names whose real DefId is currently
  unclaimed (permissive) AND whose ops scheme is wrong** (`optional`, `getModificationTime` are known
  members). For each regression, add to a "keep-permissive" set + log it; once stable, delete the
  positional assignment and the collision guard. Guard against re-drift with a unit test asserting
  every ops name resolves in lowering's builtin set. Reproduce the map: `python3 .claude/drift-builtin-lists.py` from repo root.

**EMPIRICAL (2026-07-21) — DECISIVE: the unification is a cleanup with NO score upside, MANY regressors.
Do NOT pursue it as a score lever.** Ran a full runtime-hooked bisection (env vars `BHC_DROPALL`/
`BHC_KEEP`/`BHC_DUMP` on the guard; enumerate + bisect against the real `--package-dir` full-tree
score, not standalone Builder — standalone is unfaithful because claimed/unclaimed DefId status
depends on the whole-program def map). Findings:
- Dropping the guard for *all* unclaimed builtins → **93 → 76**. The affected universe is **605 names**
  (not the 78 ops-only names — the by-name table also holds qualified `Data.Map.*`/`Data.Text.*`,
  class methods, transformer/char builtins via `register_value`/`insert_global_by_name`).
- **Name-keying the whole affected set nets ZERO gain: KEEP=all(605) → 93 = baseline.** So there is no
  subset whose name-keying raises the score above 93; the mechanism only ever loses.
- **Regressors are MANY and spread out, not one Builder culprit** (earlier "single cascade / 17 return
  for free" was WRONG): KEEP=left-303 → 83, KEEP=151 → 80, KEEP=76 → 77, KEEP=38 → 76 … i.e. dozens of
  builtins have wrong by-name schemes that only stay harmless because the collision guard never applies
  them. A single-culprit binary search is invalid here (it walked to a meaningless endpoint).

**Conclusion:** the collision guard is already extracting all the safe value from the drift; the ops
table has *many* wrong schemes that a full unification would expose. Cleaning it up is a large
correctness project (audit hundreds of schemes) with **no score payoff** — not the "highest
score-per-effort lever" it was framed as. **Redirect:** score gains come from (a) targeted curated
overrides in the `match name` block (QName/optional style, when a specific wrong scheme blocks a
module) and (b) the deeper type-mismatch cluster (the ~1000 `expected…found…` errors), NOT from the
builtin-list refactor. The drift map / scripts remain for whoever eventually does the cleanup.

**2026-07-20c — 89 → 92. SYSTEMIC fix for the builtin-list DefId drift (collision-only re-alignment).**
Root cause recap: `register_primitive_ops` registers builtin schemes at DefIds it GUESSES by index,
assuming its `ops` list mirrors bhc-lower's `builtin_funcs`; they've drifted, so a builtin's real
lowering DefId often carries an unrelated op's scheme (optional→`[Char]->Bool`, getModificationTime→
`Map String a->Maybe a`, etc.). `register_value` ALSO records every op BY NAME, so in
`register_lowered_builtins` (which has the real lowering DefId per name) we re-register builtins at
their real DefId using the name-keyed scheme. **KEY: only re-align a DefId that ALREADY carries a
scheme (a genuine collision).** The first attempt applied the name scheme to EVERY builtin def →
89→**76** (−13, 14 regressions): builtins whose real DefId is unclaimed take the permissive fresh-var
path (second pass), and forcing their often-wrong/incomplete ops-table scheme on them regresses
widely. Guarding on `lookup_def_id(id).is_some()` (collision only) → 89→**92 (+3: Class.IO, Data,
Readers.LaTeX.SIunitx), ZERO regressions**, workspace `cargo test --all-features` 2747/0. The
per-name `optional`/`getModificationTime` big-match overrides are STILL needed (their name-keyed
ops-table schemes — `[a]->[Maybe a]`, `IO String` — are themselves wrong; the big match runs after
re-alignment and overrides). NOTE: no standalone unit test — the collision is context-dependent (it
depends on the whole program's DefId assignment), so it can't be reproduced in isolation; validated
by the Pandoc integration measurement + full suite. Fuller cleanup (make the two builtin lists a
single source of truth, or fix the ops-table's wrong schemes) remains, but the collision class is now
handled generally.

**2026-07-20b — 88 → 89. `getModificationTime` mis-typed (another builtin-list-drift victim) FIXED;
`Class.PandocPure` flips.** Same root cause as `optional`: `Directory.getModificationTime`
(`DefKind::Value` builtin) picked up an unrelated `Map String a -> Maybe a` scheme via the DefId-by-index
drift; it had a correct `String -> IO String` scheme in `register_primitive_ops` but at the wrong DefId,
and wasn't in the name-based `register_lowered_builtins` match. FIX (context.rs): add `getModificationTime`
to the System.Directory arm typed `FilePath -> IO UTCTime` (UTCTime referenced by name — it's an imported
type con, unifies with `FileInfo.infoFileMTime`; the `register_primitive_ops` version's `IO String` was
wrong for real use), plus `getPermissions` (`String -> IO String`). Regression:
`bhc-driver/tests/directory_builtin_scheme.rs`. Workspace 2747/0, no regressions. **This is the 2nd
per-name patch for the builtin-list drift — confirms the whack-a-mole; the systemic fix (single source of
truth for lowering `builtin_funcs` ↔ typeck `register_primitive_ops`) would clear a whole class at once.**
Remaining 1-type-error modules: `Class.IO` (`report $ IgnoredIOError $ pack ...` — LogMessage ctor arity),
`Data` (`expected Int, found ([Char]->t)` — under-applied fn), `Writers.OOXML` (`QName{…}` external-record
construction inferred as a partial fn — distinct record-construction bug), `Parsing.Lists` (romanNumeral
do-block monad inferred `IO` not `ParsecT` — monad inference).

**2026-07-20 — 87 → 88. `optional` mis-typed as `[Char]->Bool` FIXED (the 2026-07-18d lead); cleared
ALL 37 tree-wide "found Bool" errors, +1 module (CSS flips), no regressions.** Root cause (traced via
`infer.rs` Var-lookup instrumentation): `optional` resolves to `DefId(391)`, and typeck's
`register_primitive_ops` assigns schemes to DefIds BY INDEX from an `ops` list (builtins.rs:621) that
has **drifted out of order** vs bhc-lower's `builtin_funcs` list (context.rs:344) — they diverge at
index 53 and differ in length (671 vs 343), despite the "Order MUST match" comment. So DefId 391,
which lowering uses for `optional`, gets some *other* op's `[Char]->Bool` scheme. `register_lowered_builtins`
(runs after, could override by name) didn't help: the builtin `optional` is `DefKind::Value`, the only
`optional` arm was `DefKind::StubValue`-guarded, so it hit `_ => continue` (pass 1) and pass 2 skips
already-registered DefIds. FIX (context.rs): add unqualified `"optional" | "optionMaybe"` to the
UNGUARDED parser-combinator arm (permissive `a -> b`, like `many1`/`noneOf`), which overrides the stale
scheme; removed the now-shadowed `optional` from the StubValue arm. Regression:
`bhc-driver/tests/optional_scheme.rs` (2 cases: `optional x`, `p <* optional q`). Workspace
`cargo test --all-features` **2746/0**. **The other ~9 modules with "found Bool" did NOT flip — layered
blockers underneath** (their Bool errors cleared but deeper type errors remain). **SYSTEMIC NOTE: the
lowering `builtin_funcs` ↔ typeck `register_primitive_ops` DefId-by-index drift is a latent bug — other
builtins past index 53 may be silently mistyped and masked by `register_lowered_builtins` name-matching.
A real fix would make the two lists a single source of truth; per-name overrides (as here) are the
tactical patch.**

**2026-07-18d — LEAD RESOLVED 2026-07-20 (see above): `optional` types as `Bool`, breaking ~10 parser modules.**
The recurring tree-wide `type mismatch: expected (t t), found Bool` errors (CSS, Muse, and ~8 more
parser modules) trace to the Applicative/Parsec combinator **`optional`**: standalone `g x = optional x`
(x :: Maybe Int) infers `optional x :: Bool` (plus a spurious `expected Int found Char`), so any
`p <* optional q` in a Parsec `do` fails. `optional` SHOULD be `Alternative f => f a -> f (Maybe a)`.
The `Bool` source is ELUSIVE and NOT any of: the builtins.rs `register_primitive_ops` table (proven
NOT consulted — replacing `optional`'s scheme there with `Int->Int` left the `Bool` error unchanged;
that table's `optional` was list-specialized `[a]->[Maybe a]`, also wrong but dead); `register_lowered_builtins`
(instrumented both passes — never fires for a def named `optional`, so it's not a program/interface
def); the `Char->Bool` predicate list (context.rs:2379); or the 3 known `optional` handlers
(builtins.rs:3069, context.rs:6635 `Text.Parsec.Combinator.optional`→`a->b`, context.rs:6938
Options.Applicative StubValue-guarded→`a->b`). So `optional`'s `Bool` comes from a name-based env
lookup / resolution path not yet traced — NEXT: instrument the `Expr::Var` type lookup itself (by
DefId AND by name) to see which table hands back the `Bool` scheme. This is a MULTI-MODULE lever
(~10 modules) but a real architecture dig, NOT a quick win. (Same story for `Writers.OOXML`'s single
error: `QName{…}` record construction inferred as a partial function `Text -> Maybe t -> t` — a
record-construction typing bug, also not quick.)

**2026-07-18c — OPEN LEAD (not fixed): `Writers.Docx.Table` do-block scope break.** Its 15
`unbound variable` errors (tablenum, ident, captionBlocks, colspecs, thead, tfoot, head', bodies,
foot', gridCols, …) are NOT a decl-drop — the function `tableToOpenXML` is checked but its whole
`do`-block loses scope. Mechanically reduced IN the real file (backup+restore; pandoc isn't git):
the trigger is the statement `let (Grid.Table (ident,_,tableAttr) caption colspecs _rowheads thead
tbodies tfoot) = gridTable` — deleting it restores scope to the entire rest of the block; keeping it
(even with RHS on the same line, or the nested tuple flattened to one var) breaks it. The QUALIFIED
constructor is load-bearing: swapping `Grid.Table`→unqualified `Table` changes the error to `unbound
constructor: Table` and the scope break DISAPPEARS. BUT it did NOT reproduce standalone across ~8
variants (qualified record ctor, 7 fields, positional match, `_`-prefixed var, leading stmt, dual
qualified+unqualified import) — all pass in isolation, so it needs the full real-module import/decl
environment. LOW cascade value (Docx.Table imported by only 1 module, Docx.OpenXML), so PARKED as a
lead, not a lever. Note: `import Text.Pandoc.Writers.GridTable` (unqualified, no list) does NOT bring
`Table` into scope unqualified in this module (hence the `unbound constructor: Table` when tried) —
possible related import-resolution quirk worth a look. When revisiting: bisect what module-level
context (which import / prior decl) is required to make the standalone repro fail.

**2026-07-18b — 85 → 87. NESTED `let … in` LAYOUT double-close bug fixed (real SCORE LEVER, +2, no
regressions).** Fresh triage corrected the stale notes: tree-wide `No instance for Walkable` is now
**0** (the container-instance fix cleared it), so the dominant errors are 1289 type-mismatches + 401
unbound-var + 172 unbound-ctor. Many failing modules are 1–2 errors from passing. Spot-checked the
`opts`/`emit`/`st`/`showDim`/`parStyle` unbound-variable cluster (the notes' "cascade or real
binding bug?" question): `Writers.Textile:showDim` is a `let`-bound helper `showDim dir = let toCss
… in case …`, referenced by a sibling `let` binding, yet reported unbound → the silent-drop
signature again. Reduced to `h = let a = let x = 1` / `in x` / `in a` (inner `let`'s `in` on its own
line). Root cause = **lexer layout**, `bhc-lexer/src/lib.rs` `handle_layout`: when `in` starts a line
and dedents past the inner `let`, the indentation rule closes that inner `let` (emits `}`), and then
the dedicated `in` handler *also* closed a second `let` (the outer one) — an unbalanced extra `}` that
corrupted `let a = let x = … in x in a`, so error recovery dropped the whole enclosing declaration
(hence downstream `unbound variable: showDim`, and the fn itself). This is the idiomatic
`in`-under-`let` layout, so it hit many writers. Fix: (1) track when the indentation dedent already
closed a `let` for this `in` and skip the `in` handler's close; (2) `in` at an equal-column boundary
is a `let` terminator, never a new layout item, so suppress the spurious `VirtualSemi` there too.
Regression tests: `bhc-lexer` `test_layout_nested_let_multiline_in` (virtual-brace balance) +
`bhc-parser/tests/nested_let_layout.rs` (3 cases, declaration survives). Workspace
`cargo test --all-features` **2744/0**. Pandoc **85→87 passed / 83→82 failed / 53→52 skipped**:
`Writers.Textile` and `Writers.LaTeX.Util` flipped OK, **zero regressions**. NOTE: this is the SAME
declaration-drop symptom class as the 2026-07-18 view-pattern bug and the 2026-07-02 parenthesized-
infix-operator bug — parser/lexer bugs that silently drop a decl and masquerade as `unbound variable`.
**When a clean top-level/where/let binding reports `unbound variable` at its use site with no error on
its own body, suspect a parse/layout drop, not a resolver bug.** NEXT: the remaining
`opts`/`emit`/`parStyle`/`ident` unbound clusters (ICML 24, Docx.Table 15, etc.) — check whether more
are decl-drops (reduce one) vs genuine RecordWildCards/where locals; and the 1-type-error modules
(`Class.IO`, `Class.PandocPure`, `CSS`, `Writers.OOXML`) are mostly external/stubbed-type mistypings
(e.g. OOXML `QName{…}` record construction inferred as a partial function; PandocPure
`getModificationTime` typed like `Map.lookup`).

**2026-07-18 — the "single-local-drop lowering bug" was actually a PARSER bug: view patterns as
tuple/list ELEMENTS dropped the whole enclosing declaration. SCORE-NEUTRAL (85→85), no regressions.**
Triaged the current 1-error `lowering failed` modules; the only genuine local-drop candidates were
`DokuWiki:splitInterwiki` and `Citeproc.Data:biblatexLocalizations` (the rest — `dataFiles'`,
`MathMLEntityMap.getUnicode`, `TM.writeTypst`, `ensureValidXmlIdentifiers` — are TH splices /
qualified cross-module imports / cascade). `biblatexLocalizations` is also a TH `$(embedDir ..)`.
`splitInterwiki` was the real one: a clean top-level fn reported `unbound variable: splitInterwiki`
at its use site with NO error on its own body — the classic silent-drop signature. Reduced to a
30-line self-contained repro, then minimised: the trigger is a **view pattern nested inside a tuple
(or list) pattern** — `(l, T.uncons -> Just ('>', r))` — where the enclosing fn is **referenced by
any other top-level binding**. Root cause is in the **parser**, not lowering: view-pattern parsing
(`expr -> pat`) lived only in `parse_paren_pattern` and assumed the view pattern was the *sole*
paren content (it did `expect(RParen)` right after the result pattern). As a tuple element the view
is parsed by the general `parse_pattern`, which stops at `->`; the tuple loop then sees `->` instead
of `,`/`)`, `expect(RParen)` fails, and error recovery **silently discards the entire enclosing
`FunBind`** (confirmed: `collect_module_definitions` saw `[TypeSig, TypeSig, FunBind(caller)]` — the
`zog=` binding was gone). Downstream, every use of the dropped fn reports `unbound variable`. (Why it
looked name-dependent while bisecting: `f` is a registered stub in `context.rs`, so a fn coincidentally
named `f` resolved to the stub and masked the drop.) Fix (`bhc-parser/src/pattern.rs`): factored out
`parse_pattern_or_view` (the existing simple + applied view logic, minus the `expect(RParen)`) and use
it for every tuple element (`parse_paren_pattern`) and list element (`parse_list_pattern`); standalone
`(e -> p)` still returns the bare `Pat::View`. Regression tests: `bhc-parser/tests/view_pattern_in_tuple.rs`
(5 cases: 2nd/1st tuple element, list element, applied view `f k -> p`, standalone). Full workspace
`cargo test --all-features`: **2740/0** (needs `LIBRARY_PATH=/opt/homebrew/opt/openblas/lib` for the
driver lib-test link). Pandoc: **85/83/53, byte-identical module-status set vs before** — DokuWiki
went `lowering failed: 1` → `type checking failed: 116 errors` (LAYERED BLOCKER: the reader machinery
has deep type errors underneath; parser fix is necessary-not-sufficient). Correct general fix
regardless. NEXT: the remaining near-passing modules are TH/qualified-import/cascade, not local drops;
the real Pandoc score lever is still the `Walkable a [b]` instance-chaining / writer-module type-error
cluster (see P1/P3 below), not lowering.

**2026-07-11 — 82 → 85. Cross-module PATTERN SYNONYM export was the real blocker on the two
1-error modules (MediaWiki, Texinfo), NOT a local drop.** Both failed on `unbound constructor:
SimpleFigure` — a bidirectional pattern synonym in pandoc-types `Text.Pandoc.Definition`
(`pattern SimpleFigure attr cap tgt <- Para [Image .. (isFigureTarget -> Just tgt)] where ..`).
Root cause: pattern synonyms are bound locally (resolve.rs) and expanded inline in the DEFINING
module, but were never propagated to a module's **exports**, so importers saw them as unbound. Inline
expansion can't work cross-module anyway — the RHS uses the *unexported* helper `isFigureTarget`, so
importers must treat the synonym **opaquely** (as GHC does). Fix: export a pattern synonym as an
opaque constructor `forall f1..fn. f1 -> .. -> fn -> Result`, where `Result` is inherited from the
RHS head constructor's data type (`Para` ⇒ `Block`); fields stay polymorphic. `check` stops after
typeck (no codegen), so no real constructor tag is needed. Landed in three export paths:
`bhc_lower::loader::collect_exports` (+ `pattern_synonym_con_shape`), and the driver's
`build_module_exports_from_hir` (primary; pulls synonyms from `LowerContext::pattern_synonyms()`
since HIR has no synonym item) and `build_module_exports_from_ast`. Result: **MediaWiki OK, Texinfo
OK, XWiki SKIPPED→OK** (imports MediaWiki). Regression test
`test_cross_module_pattern_synonym_resolves` (bhc-driver integration). Full workspace `cargo test
--all-features`: **2733/0**. Score **85/82/54**.

**2026-07-11b — infix operator definition `x . y = ...` operand-drop FIXED (general parser bug;
score-neutral for pandoc).** `arrow2` (ODT.Arrows.State) was `arrow2 . arrow1 = ArrowState $ ..` —
an infix definition of `.`. Root cause: `.`/`*`/`%` lex as their own tokens (`Dot`/`Star`/`Percent`),
but `is_infix_var_op_start`/`parse_infix_op` (bhc-parser/src/decl.rs) only recognized
`Operator`/`Backtick`, so `x . y = ..` missed the infix-definition branch and the FIRST operand was
never bound (surfaced as `unbound variable` when the body used it). Fix: recognize `Dot`/`Star`/
`Percent` in both (mirrors the set `parse_var_or_op` already accepts). Minimal repro `x . y = run x`
now OK. Regression test `test_infix_definition_of_special_token_operator` (bhc-parser). **This did NOT
move the score:** ODT.Arrows.State's lowering error is gone but it now hits `type checking failed: 15
errors` (deeper blockers), and ODT.Generic.XMLConverter went SKIPPED→FAILED (now checkable, still
fails). Score unchanged at 85 (85/83/53). Correct general fix regardless; necessary-not-sufficient
for that module cluster.

**NEXT (remaining 1-error modules):** mostly **TH / qualified-import**, out of the plain
local-drop scope: `BakedIn:dataFiles'` (RHS is a TH splice `$(embedFile ..)`), `Typst:TM.writeTypst`
+ `Org.Inlines:MathMLEntityMap.getUnicode` (qualified imports of failing/absent modules),
`TEI:ensureValidXmlIdentifiers` (cascade — defined in Writers.Shared which itself fails). Also
`DokuWiki:splitInterwiki`, `Citeproc.Data:biblatexLocalizations` — verify whether where/let locals
or cascade. The ODT.Arrows cluster needs its 15 type errors triaged (arrow instances / MPTC), not a
single-local fix.

**2026-07-07 — mechanical stub sweep done; it is NOT a score lever (still 82/84/55).** Added
missing stdlib exports (Data.Sequence breakl/…, Control.Monad `<$!>`, Text.XML.Light.Output
ppc*/useShortEmptyTags/defaultConfigPP, TagSoup isTagOpenName/isTagCloseName; commit ecdde7e).
They resolve their names but **flip ZERO modules**: every failing module has LAYERED blockers —
clearing the top unbound name just reveals a type error or another unbound name beneath (e.g.
`CSS` went `unbound isTagOpenName` → `type checking failed: 1`). **THE REAL LEVER (highest
value found): ~10 modules are ONE unbound *local* away from passing** — `MediaWiki:n`,
`Typst:st`, `Texinfo:v'`, `JATS.References:bodies/foot'`, `TEI:nestle`, `ODT.Arrows.Utils:dataFiles'`,
`Citeproc.Data:arrow2`, plus DocBook/Ms whole-equation local drops (`opts/id'/lvl/divattrs/hclasses`).
These all smell like ONE equation/where/let-lowering bug that drops locals; my minimal repros of the
obvious constructs (string-literal cons pattern in a tuple) PASS in isolation → emergent, needs
per-module reduction like the Walk bugs. **NEXT: hunt that single-local-drop lowering bug** (pick
`MediaWiki` or `Texinfo` — 1 error each — and reduce which construct unbinds the local); cracking it
could flip several near-passing modules at once. The stub sweep is exhausted as a lever.

**Current measurement (reconfirmed 2026-07-04, Pandoc 3.6.4):**

```
bare src:                 79 passed, 89 failed, 53 skipped  (of 221 modules)
+pandoc-types package-dir: 82 passed, 84 failed, 55 skipped
```

Reproduce (Pandoc tree `/Users/zara/Development/pandoc-3.6.4`, pandoc-types
`/Users/zara/Development/pandoc-types-1.23.1`):

```
BHC=<bhc-repo>/target/debug/bhc
PT=/Users/zara/Development/pandoc-types-1.23.1/src
P=/Users/zara/Development/pandoc-3.6.4/src
LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21 $BHC check "$P" 2>/dev/null | tail -1               # 79
LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21 $BHC --package-dir "$PT" check "$P" 2>/dev/null | tail -1  # 82
```

The `--package-dir <pandoc-types>/src` form is the recommended invocation: it binds
the pandoc-types `Text.Pandoc.Definition` constructor family (`Str`/`Div`/`Para`/
`Align*`/…) for the whole tree. The +3 gain is capped because `Walk` and `JSON`
(pandoc-types modules that many Pandoc modules import) still fail → cascade skips.

> **Note:** this supersedes the previous baseline ("10 passed / 195 failed").
> That figure and its "Category A" workplan are obsolete — most of the gap has
> since closed (stub mechanism, deriving, transformers, containers, etc.).

---

## What's already handled

- **Qualified builtin imports** (`T.pack`, `Map.insert`, `Set.member`, …) —
  RESOLVED. The lowering stub mechanism (`LowerContext::with_builtins`) resolves
  unknown qualified names from builtin modules, emitting
  `warning: stub function ... used`. (A `builtin_module_exports` table in
  `check_files_ordered` was tried and confirmed a NO-OP — do not redo it.)
- **Skipped modules (52):** modules whose imports can't be satisfied (a
  dependency module failed, or an unvendored package). Many will resolve for
  free as their dependencies start passing.

---

## Current failure breakdown

From the stderr of the check run, by frequency:

| Cause | ~Count | Notes |
|-------|--------|-------|
| Type-mismatch cascade (`expected … found …`) | ~1000 | Mostly downstream of unbound names typed as fresh vars; shrinks as unbound names are fixed |
| `unbound variable` | ~418 | See below |
| `unbound constructor` | ~216 | Dominated by external-package ADTs |
| Type-error summaries | ~51 | One per failing-typecheck module |
| `No instance` | ~16 | `Num Text` (11), `Num [Int]` (5) — numeric-literal defaulting / typeck |

### Top unbound constructors (external-package ADTs, not in `src/`)

`Div`(22), `FancyVal`(9), `EGrouped`(6), `AlignLeft`(5), `Accent`(4),
`TUnder`/`TOver`(4), `OrderedList`(3), `LExact`(3), `ItemId`(3), `AlignRight`(3),
`Str`/`Space`/`VBoolean`(2), plus `Jira.*`(Row/Parameter/User/Space/HeaderCell/BodyCell).

The `Div`/`Str`/`Space`/`Align*`/`OrderedList` family is **pandoc-types**
(`Text.Pandoc.Definition`) — a separate package not vendored under `src/`. This
is the single biggest cascade: every reader/writer that constructs or matches
these fails. `FancyVal`/`EGrouped`/`Accent`/`TUnder`/`TOver` are internal reader
types (cascade from their defining module failing).

### Top unbound variables

`opts`(23), `toValue`(18), `emit`(17), `!`(17), `addAttrs`(13), `html5`(9),
`ensureValidXmlIdentifiers`(9), `B.<>`(9), `parStyle`(8), `st`/`classes`(7),
`slideVariant`/`format`/`customAttribute`/`contents`/`anyToken`(6),
`showDim`/`notes`/`ident`/`cs`(4), `txt`/`tokenItalic`(3).

- `B.<>` — qualified operator (Text.Pandoc.Builder); qualified-operator import
  resolution worth a check.
- `opts`/`st`/`classes`/`contents`/`ident`/`cs`/`txt` — look like local bindings
  (function params / where / RecordWildCards). Could be cascade (the enclosing
  def already failed) OR a real name-resolution bug. **Spot-check one** before
  assuming cascade — a systematic binding-resolution bug here would unblock many.

---

## Workplan (priority order)

### P1 — pandoc-types (`Text.Pandoc.Definition`) — IN PROGRESS (2026-07-02)
Approach that works: `bhc check --package-dir <pandoc-types>/src <pandoc>/src`
(package-dir modules resolve imports but aren't scored). Fetched pandoc-types
1.23.1 from Hackage to `/Users/zara/Development/pandoc-types-1.23.1`.

Blockers found + resolved so this loads:
- `Definition` imports `Paths_pandoc_types` (Cabal autogen module) → stubbed a
  minimal `Paths_pandoc_types.hs` (`version = makeVersion [1,23,1]`). **General
  issue**: many Hackage pkgs import `Paths_<pkg>`; a real fix is auto-synthesizing
  it in bhc.
- `Definition`/`Builder`/`Generic` used unstubbed external symbols (aeson
  `toEncoding`/`TaggedObject`, `Seq.dropWhileL/R`, `Map.fromAscList`/
  `foldMapWithKey`, `Traversable.fmapDefault`, `everywhere'`) → **added to the
  stub lists** (bhc-lower, commit 7ca995e).
- **Real parser bug fixed** (commit 30d519a): `(Many xs) <> (Many ys) = … where …`
  (infix operator def with parenthesized operands) was misparsed as a pattern
  binding, unbinding the body's `where`/`let` vars. This alone took Builder from
  22 lowering errors → OK.
- Curated the vendored `Definition.hs` to ADT-only (removed the aeson TH splice +
  hand-written JSON instances + `pandocTypesVersion`) since bhc can't yet load
  TH/aeson-heavy source. **Definition/Builder/Generic now load OK.**

**Result: pandoc 79 → 82 passed (89 → 84 failed).** Modest because the jump is
capped by the rest of the cluster: `Walk` (13 type errors) and `JSON` (4 lowering
errors) still fail, and both are widely imported → cascade skips.

`Walk` investigation — **prior fixes + exhaustive 2026-07-04 drill (10 synthetic repros, ALL PASS).**

Earlier (2026-07-02/03): the `(t b)` catch-all head not matching concrete `[b]` was
**FIXED** (commit 3fca571, `types_match_with_subst` App-vs-List). Typed Map stubs
(`fromAscList`/`fromDistinctAscList`/`foldMapWithKey`) added (commit 46c94c3) — but
neither changed Walk's 13 errors.

**2026-07-04 — reproduced the state (82/221 with `--package-dir <pandoc-types>/src`;
the `Str`/`Div`/`Para`/`Align` family now BINDS, gone from the unbound list) and
drilled Walk hard. The 13 errors: 3× `expected MetaValue, found [Block]` + 10×
`No instance for Walkable Block [Block]` cascade. Correction to the error location:
the flagged spans are `walkMetaValueM'` (MetaMap/MetaList helper: `M.fromAscList`/
`M.toAscList`/`mapM`) and `queryMetaValue'` (`M.foldMapWithKey`) — NOT the
`MetaBlocks bs` case.**

Ruled out via 10 minimal repros that each PASS `bhc check` in isolation:
1. recursive `instance Walkable a b => Walkable a [b]`;
2. `(t b)`-head `instance (Foldable t, Traversable t, Walkable a b) => Walkable a (t b)`;
3. polymorphic given-derivation (`Walkable a Block` given ⟹ `Walkable a [Block]` wanted);
4. the real 3-method class (`walk` default + `walkM` + `query`) with `walkMetaValueM`'s
   exact signature `(Walkable a [Block], Walkable a [Inline], …)`;
5. the use-site `instance Walkable Block MetaValue where walkM = walkMetaValueM`
   (instantiates `a=Block`, forcing `Walkable Block [Block]` as an instance not a given);
6. overlapping self-list instances `{-# OVERLAPPING #-} Walkable [Block] [Block]` /
   `[Inline] [Inline]` alongside `(t b)`;
7. direct overlap: `Walkable [Inline] [Inline]` matched by BOTH `(t b)` and the specific
   OVERLAPPING instance, with a use forcing resolution;
8. cross-module imported constructors (2-file repro, `MetaValue` imported via `--package-dir`);
9/10. the exact `walkMetaValueM'`/`queryMetaValue'` code with `MetaValue = MetaMap (Map Text
   MetaValue) | MetaList [MetaValue] | MetaBlocks [Block]` + the real Data.Map stubs.

Also confirmed **bhc runs CPP** (`#define OVERLAPS {-# OVERLAPPING #-}` expands correctly).

**Conclusion (2026-07-04): the trigger is IRREDUCIBLY the full-module combination** —
superseded in part by the 2026-07-06 root-cause below.

**2026-07-06 — brute-reduced the real `Walk.hs` and split the 13 errors into TWO
independent bugs. ONE IS FIXED.**

Reproduce fast (checks Walk as scored target with Definition+Paths available):
`bhc --package-dir <pandoc-types>/src check <pandoc-types>/src/Text/Pandoc/Walk.hs`
→ was 13 errors (10× `No instance Walkable X [Y]` + 3× `expected MetaValue, found [Block]`).

- **BUG 1 — `Walkable a [b]` (a≠b) instance resolution — FIXED.** Root cause found by
  reading the solver, not more repros: the generic `instance (Foldable t, Traversable t,
  Walkable a b) => Walkable a (t b)` matched the head fine (`types_match` already handles
  App-vs-List), but discharging its **context** with `t ~ []` needs `Foldable []` /
  `Traversable []`, and `is_builtin_instance` (context.rs) knew **no** Foldable/Traversable/
  Functor/… instances at all → context failed → generic instance rejected → spurious
  `No instance for Walkable Inline [Block]`. (Also why the earlier "add an explicit
  `Walkable a [b]` instance" experiment couldn't help: `resolve_instance_multi` returns the
  FIRST head-match with **no backtracking** — the generic instance shadows any added one and
  its failing context kills resolution.) Fix: teach `is_builtin_instance` that the standard
  container constructors (`[]`,`Maybe`,`Either`,`Map`,`IntMap`,`Seq`,`NonEmpty`,`Identity`,
  `IO`,`(,)`) satisfy `Functor`/`Foldable`/`Traversable`/`Applicative`/`Monad`/… via a new
  `container_head_name` helper. **Result: tree-wide `No instance Walkable` 10 → 0.**
  Regression tests: `test_container_head_name`, `test_higher_kinded_container_instances`.

- **BUG 2 — the 3× `expected MetaValue, found [Block]` — FIXED (2026-07-06). It was a PARSER
  bug, not a solver bug.** Instrumented `unify`/`bind_var`/the `Var` scheme lookup and traced
  the `[Block]` back: the body's `query` reference resolved to **DefId of an instance method**
  (`instance {-# OVERLAPPING #-} Walkable [Block] [Block]`'s `query`, scheme `([Block]->c)->[Block]->c`,
  no class constraint) instead of the polymorphic class method — so `[Block]` flowed straight in.
  Root: `query` was `bind_value`'d 4× in module scope; the extra 3 were the `query` methods of the
  three `instance OVERLAPS …` instances. `parse_instance_decl_with_doc` did **not consume the
  `{-# OVERLAPPING #-}` pragma** after `instance`; the head parse derailed and error recovery
  reparsed each overlapping instance's `where` methods as **top-level `FunBind`s**, shadowing the
  class method. (Only the `OVERLAPS`-pragma instances leaked; plain `instance … where` was fine.)
  Fix: skip pragma tokens right after `expect(Instance)` in `crates/bhc-parser/src/decl.rs`.
  Regression test: `crates/bhc-parser/tests/overlapping_instance.rs`. **Result: real unmodified
  `Walk.hs` now PASSES `bhc check` (0 errors).** Full workspace `cargo test --all-features`: 2732/0.

**IMPORTANT re-measurement (2026-07-06): fixing BOTH bugs did NOT move the pandoc score (still
82/84/55).** Walk.hs passing standalone doesn't help because as a `--package-dir` module its
typecheck result doesn't gate scored importers (exports register regardless), and both bugs lived
in the pandoc-types package-dir modules, not the scored pandoc set. Only ONE scored module
(`Text/Pandoc/Class/PandocMonad.hs`) and pandoc-types `Builder.hs` use the overlap pragma. So the
real gate on the scored pandoc modules is ELSEWHERE — the `unbound variable`/`unbound constructor`
cascades in the writer/reader modules (the P2/P3 territory), NOT Walk. **NEXT: stop treating Walk
as the keystone; re-triage the scored failing modules directly** (pick a scored FAILED module,
read its first few real errors) to find what actually gates the 84 failures. Both compiler fixes
here are correct and general regardless of pandoc's score.

### P2 — Cascade-critical internal modules
Once pandoc-types resolves, re-run and find which remaining failures are
internal cascade (e.g. the modules defining `FancyVal`/`EGrouped`/`Accent`).
Fix the upstream module; downstream + skipped modules follow.

### P3 — Resolver spot-check (`opts`/`st`/`B.<>`/`!`) — DONE (2026-07-06): NO independent bug; collapses into P1
Probed the top `unbound variable` (`ensureValidXmlIdentifiers`, 4×; also
`showDim`, and the whole writer-module cluster `opts`/`classes`/`contents`/…).
**Finding: not a name-resolution bug — it is cascade from the Walkable
instance-resolution failure (same root cause as P1).**

- `ensureValidXmlIdentifiers` is *defined* in `Text/Pandoc/Writers/Shared.hs`
  and imported by EPUB/TEI/ICML/ODT/HTML/DocBook/FB2. Those importers report it
  `unbound` **because Shared.hs itself FAILS**: `Writers.Shared FAILED: type
  checking failed: 21 errors`, driven by `No instance for Walkable {Inline,Block}
  [{Inline,Block}]`. Name resolution is fine; the exporting module never
  type-checks, so its exports don't bind downstream.
- Whole-tree total is only **10** `No instance for Walkable X [Y]` errors, all of
  the canonical list shapes: `Walkable Inline [Inline]/[Block]`,
  `Walkable Block [Inline]/[Block]`, `Walkable [Inline] [Inline]/[Block]`. Each
  should resolve via `instance Walkable a b => Walkable a [b]` (e.g.
  `Walkable Inline [Block]` ⇒ `Walkable Inline Block`, a real pandoc-types
  instance). bhc fails to chain this in the full-module instance environment —
  exactly the P1 Walk trigger, and it also surfaces in the smaller **Shared.hs**
  (21 errors) which may be an easier reduction target than `Walk.hs`.

**Net: P3 is not a separate lever. The single dominant Pandoc blocker is
`Walkable a [b]` (a≠b) instance chaining in the full-module context. Next
concrete step = brute-reduce `Writers/Shared.hs` (21 errors) or `Walk.hs` to the
minimal instance set that still fails, per P1.**

### P4 — `No instance for Num Text` / `Num [Int]`
Investigate the 16 numeric-literal-defaulting type errors — likely a typeck
interaction with OverloadedStrings / list literals in a numeric context.

### P4 — `No instance for Num Text` / `Num [Int]`
Investigate the 16 numeric-literal-defaulting type errors — likely a typeck
interaction with OverloadedStrings / list literals in a numeric context.

---

## Measurement log

| Date | passed | failed | skipped |
|------|--------|--------|---------|
| (old, stale) | 10 | 195 | 16 |
| 2026-06-27 | 79 | 90 | 52 |
| 2026-07-02 (bare `src`) | 79 | 89 | 53 |
| 2026-07-02 (+pandoc-types package-dir, Def/Builder/Generic loading) | 82 | 84 | 55 |
| 2026-07-04 (reconfirmed bare `src`) | 79 | 89 | 53 |
| 2026-07-04 (reconfirmed +pandoc-types package-dir) | 82 | 84 | 55 |
| 2026-07-11 (+cross-module pattern synonym export; MediaWiki/Texinfo/XWiki) | 85 | 82 | 54 |
| 2026-07-18 (+view-pattern-in-tuple/list parser fix; DokuWiki past lowering) | 85 | 83 | 53 |
| 2026-07-18b (+nested `let…in` layout double-close fix; Textile/LaTeX.Util) | 87 | 82 | 52 |
| 2026-07-20 (+`optional` scheme fix; CSS flips, 37 "found Bool" cleared) | 88 | 81 | 52 |
| 2026-07-20b (+`getModificationTime` scheme fix; PandocPure flips) | 89 | 80 | 52 |
| 2026-07-20c (+collision-only builtin DefId re-alignment; +3 modules) | 92 | 77 | 52 |
