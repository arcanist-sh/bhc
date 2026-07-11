# Pandoc `bhc check` — Current State & Workplan

**Goal:** `bhc check` succeeds on Pandoc's library modules (excluding Template Haskell).

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
