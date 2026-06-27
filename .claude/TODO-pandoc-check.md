# Pandoc `bhc check` — Current State & Workplan

**Goal:** `bhc check` succeeds on Pandoc's library modules (excluding Template Haskell).

**Measurement (2026-06-27, Pandoc 3.6.4):**

```
79 passed, 90 failed, 52 skipped  (of 221 modules)
```

Reproduce from the Pandoc source tree (`/Users/zara/Development/pandoc-3.6.4`):

```
LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21 \
  <bhc-repo>/target/debug/bhc check src 2>err.log | tail -1
```

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

### P1 — pandoc-types (`Text.Pandoc.Definition`) — HIGHEST LEVERAGE
Provide the real `Inline`/`Block`/`Pandoc`/`Meta`/`Attr`/`Format`/`ListNumber*`/
`Alignment`/`MathType`/`QuoteType`/`Citation*`/`Caption`/`Row`/`Cell`/… ADTs so
their constructors resolve for both construction and pattern matching (correct
arities required). Either vendor the pandoc-types source as extra `.hs` modules
in the check set, or hand-author a stub module. Re-measure after — expect a
large jump as reader/writer modules unblock.

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
