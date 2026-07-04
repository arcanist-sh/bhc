# Pandoc `bhc check` — Current State & Workplan

**Goal:** `bhc check` succeeds on Pandoc's library modules (excluding Template Haskell).

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

**Conclusion: the trigger is IRREDUCIBLY the full-module combination** of ~40
specific/overlapping `Walkable` instances + the helper-fn constraint signatures — it
does not survive extraction. The only remaining path is **brute mechanical reduction
of the real `Walk.hs`**: bisect the instance blocks (`(t b)` @L140; specific/overlapping
insts @L146–407; helper fns `walkInlineM`/`walkBlockM`/`walkMetaValueM`/`queryMetaValue`
@L416–668) — deleting groups until it passes, keeping helper-fn deps satisfied (e.g.
`walkInlineM`'s context needs the `Walkable a Citation` instances). **Do NOT re-derive
synthetic repros — they are exhausted.** Start a fresh focused session at the brute
reduction. NEXT decision: brute-reduce Walk (open-ended) vs pivot to Walk-independent
modules (P2) vs reassess.

### P2 — Cascade-critical internal modules
Once pandoc-types resolves, re-run and find which remaining failures are
internal cascade (e.g. the modules defining `FancyVal`/`EGrouped`/`Accent`).
Fix the upstream module; downstream + skipped modules follow.

### P3 — Resolver spot-check (`opts`/`st`/`B.<>`/`!`)
Pick one module with an `unbound variable: opts`-style error that *should* be
locally bound and confirm whether it is a genuine name-resolution bug
(qualified operators, where-clause scoping, RecordWildCards) vs cascade.

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
