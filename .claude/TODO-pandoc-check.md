# Pandoc `bhc check` — Path from 10 to 221

**Created:** 2026-03-01
**Baseline:** 10 passed, 195 failed, 16 skipped (of 221 modules)
**Command:** `bhc -I /tmp/pandoc-smoke/src check /tmp/pandoc-smoke/src/`
**Commits:** `342b274` (multi-module check), `a4704f4` (tuple constructor fix)

---

## Passing Modules (10)

Asciify, Base, Char, Class, CommonMark, Namespaces, Pandoc, Parsing, RoffChar, Types

## Skipped Modules (16)

Need truly external packages (Citeproc, Typst, TeXMath, Data.Aeson, Data.FileEmbed,
Text.DocLayout, etc.). These are out of scope until the package system compiles
Hackage packages. Will remain skipped.

---

## Failure Analysis

3957 unique unbound names across 195 failing modules. Categorized by root cause:

### Category A: Qualified imports from builtin stubs (~1600 occurrences)

Builtin modules (Data.Text, Data.Map, Data.Set, etc.) are registered in the
builtin set so imports don't fail, but they export **nothing** — so qualified
references like `T.pack` or `Set.member` are unresolved.

**Top qualified unbound names:**

| Name | Count | Source module | Qualifier |
|------|-------|---------------|-----------|
| `T.pack` | 449 | Data.Text | `T` |
| `T.unpack` | 226 | Data.Text | `T` |
| `T.singleton` | 92 | Data.Text | `T` |
| `T.uncons` | 82 | Data.Text | `T` |
| `T.isPrefixOf` | 63 | Data.Text | `T` |
| `T.strip` | 42 | Data.Text | `T` |
| `T.replace` | 25 | Data.Text | `T` |
| `T.cons` | 22 | Data.Text | `T` |
| `T.isInfixOf` | 21 | Data.Text | `T` |
| `B.space` | 84 | Text.Pandoc.Builder | `B` |
| `B.text` | 48 | Text.Pandoc.Builder | `B` |
| `B.divWith` | 48 | Text.Pandoc.Builder | `B` |
| `B.str` | 41 | Text.Pandoc.Builder | `B` |
| `B.plain` | 34 | Text.Pandoc.Builder | `B` |
| `Set.member` | 26 | Data.Set | `Set` |
| `BL.fromStrict` | 25 | Data.ByteString.Lazy | `BL` |
| `UTF8.toText` | 37 | Text.Pandoc.UTF8 | `UTF8` |
| `Seq.singleton` | ~10 | Data.Sequence | `Seq` |

**Fix:** When a module imports `import qualified Data.Text as T`, and
`Data.Text` is a builtin module, the lowering pass must fabricate exports
for known functions of that module. This requires a "builtin module export
table" in the driver or lower context that maps module names to their
known exported symbols.

**Impact:** HIGH — fixing Data.Text alone unblocks ~900 errors. Fixing the
top 5 modules (Data.Text, Data.Set, Data.Map, Data.ByteString.Lazy,
Data.Sequence) could unblock ~1200 errors.

### Category B: External package ADT constructors (~2500 occurrences)

Constructors from packages that aren't compiled:

| Constructor | Count | Package |
|-------------|-------|---------|
| `Str` | 246 | pandoc-types (Text.Pandoc.Definition) |
| `Para` | 190 | pandoc-types |
| `Plain` | 162 | pandoc-types |
| `Div` | 149 | pandoc-types |
| `Space` | 124 | pandoc-types |
| `Header` | 107 | pandoc-types |
| `Span` | 93 | pandoc-types |
| `Pandoc` | 91 | pandoc-types |
| `Format` | 90 | pandoc-types |
| `Image` | 84 | pandoc-types |
| `Link` | 72 | pandoc-types |
| `LineBreak` | 72 | pandoc-types |
| `TagOpen` | 110 | tagsoup (Text.HTML.TagSoup) |
| `Tok` | 89 | internal (Text.Pandoc.Readers.LaTeX.Types) |
| `Elem` | 89 | xml (Text.XML.Light) |
| `Lang` | 102 | internal (Text.Pandoc.Translations.Types) |
| `Option` | 106 | internal or external |

**Fix:** Two-part:
1. **pandoc-types stub**: Create a fake `Text.Pandoc.Definition` module (or
   `.bhi` interface) that exports the `Pandoc`, `Block`, `Inline`, `Meta`,
   `Attr`, etc. ADTs. This is a finite set (~30 types) that can be hand-written.
2. **Cross-module cascade**: Many of these constructors come from internal
   Pandoc modules that fail to compile. As categories A and C are fixed,
   these modules may start passing and exporting their types to downstream.

**Impact:** HIGH — but depends on A being fixed first (cascade effect).

### Category C: Operators from external packages (~1000 occurrences)

| Operator | Count | Source |
|----------|-------|--------|
| `$$` | 496 | Text.DocLayout (pretty-printing combinator) |
| `.=` | 130 | Data.Aeson (JSON key-value) |
| `.:?` | 87 | Data.Aeson |
| `.!=` | 64 | Data.Aeson |
| `>>^` | 57 | internal arrow operators |
| `<?>` | 33 | Text.Parsec (error labeling) |
| `~==` | 30 | Text.HTML.TagSoup |
| `$>` | 23 | Data.Functor |
| `$+$` | 21 | Text.DocLayout |
| `<+>` | 17 | Text.DocLayout |

**Fix:**
- `$$`, `$+$`, `<+>` from Text.DocLayout — needs DocLayout stub or package
- `.=`, `.:?`, `.!=` from Data.Aeson — needs Aeson stub or package
- `$>` from Data.Functor — should be in BHC builtins (it's base)
- `<?>` from Text.Parsec — needs parsec stub or package

**Impact:** MEDIUM — `$$` alone is 496 occurrences but all in writer modules.

### Category D: Parsec/megaparsec functions (~1200 occurrences)

| Function | Count | Module |
|----------|-------|--------|
| `many1` | 174 | Text.Parsec.Combinator |
| `manyTill` | 153 | Text.Parsec.Combinator |
| `lookAhead` | 140 | Text.Parsec.Combinator |
| `notFollowedBy` | 140 | Text.Parsec.Combinator |
| `skipMany` | 138 | Text.Parsec.Combinator |
| `char` | 542 | Text.Parsec.Char |
| `string` | 238 | Text.Parsec.Char |
| `getState` | 164 | Text.Parsec.Prim |
| `updateState` | 132 | Text.Parsec.Prim |
| `spaceChar` | 153 | Text.Pandoc.Parsing (re-export) |
| `newline` | 129 | Text.Parsec.Char |

**Fix:** Pandoc's `Text.Pandoc.Parsing` re-exports most parsec functions.
If Text.Pandoc.Parsing can be made to pass (it currently does pass!), then
the issue is that downstream modules import parsec directly too. Need parsec
stubs or to compile parsec from source.

**Impact:** MEDIUM-HIGH — but parsec is a real library with complex internals.
Stubs may be more practical than full compilation initially.

### Category E: Cascade failures from internal modules (~800 occurrences)

Functions from Pandoc modules that themselves fail, so their exports
aren't available downstream:

| Function | Count | Defined in |
|----------|-------|-----------|
| `literal` | 547 | Text.Pandoc.Shared (fails) |
| `mknode` | 417 | Text.Pandoc.XML (fails) |
| `tshow` | 275 | Text.Pandoc.Shared |
| `defField` | 210 | Text.Pandoc.Shared |
| `blankline` | 206 | Text.DocLayout (external) |
| `report` | 157 | Text.Pandoc.Class (fails) |
| `isEnabled` | 149 | Text.Pandoc.Options (fails) |
| `vcat` | 148 | Text.DocLayout (external) |
| `inTags` | 130 | Text.Pandoc.XML |
| `nullAttr` | 139 | Text.Pandoc.Definition (external) |

**Fix:** These are the domino effect. As upstream modules are fixed
(particularly Text.Pandoc.Shared, Text.Pandoc.XML, Text.Pandoc.Class,
Text.Pandoc.Options), their exports propagate and downstream modules
clear automatically.

**Impact:** HIGHEST — fixing 5-10 key upstream modules could clear
50+ downstream modules.

### Category F: Type checking failures (7 modules)

These modules pass lowering but fail type checking:

| Module | Errors | Notes |
|--------|--------|-------|
| Future | 1 | Closest to passing |
| Environment | 2 | |
| Symbols | 2 | |
| Standards | 2 | |
| Macros | 2 | |
| Fallible | 3 | |
| Types | 12 | Text.Pandoc.Readers.Docx.Parse.Styles.Types |

**Fix:** Need to investigate each individually. These are likely type
inference gaps (missing instances, unsupported type features, etc.)

**Impact:** LOW in count but important signal — these show real typeck bugs.

---

## Workplan: Priority Order

### P1: Builtin module export tables (Category A) — HIGHEST IMPACT

Make builtin modules export their known functions so qualified imports
resolve. Focus on the modules that appear most in Pandoc:

1. **Data.Text** — `pack`, `unpack`, `singleton`, `uncons`, `isPrefixOf`,
   `strip`, `replace`, `cons`, `isInfixOf`, `null`, `intercalate`,
   `toLower`, `toUpper`, `splitOn`, `breakOn`, `words`, `unwords`,
   `lines`, `unlines`, `any`, `all`, `filter`, `map`, `concatMap`,
   `head`, `tail`, `length`, `drop`, `take`, `dropWhile`, `takeWhile`,
   `append`, `snoc`, `empty`, `foldl'`, `foldr`

2. **Data.Set** — `member`, `fromList`, `toList`, `insert`, `delete`,
   `union`, `intersection`, `difference`, `null`, `size`, `singleton`,
   `empty`, `map`, `filter`, `notMember`, `isSubsetOf`

3. **Data.Map / Data.Map.Strict** — `lookup`, `insert`, `delete`,
   `fromList`, `toList`, `toAscList`, `empty`, `singleton`, `member`,
   `map`, `mapWithKey`, `filter`, `filterWithKey`, `union`, `unionWith`,
   `intersection`, `intersectionWith`, `difference`, `keys`, `elems`,
   `null`, `size`, `foldlWithKey'`, `foldrWithKey`, `adjust`, `alter`,
   `findWithDefault`, `mapKeys`, `insertWith`

4. **Data.ByteString.Lazy** — `fromStrict`, `toStrict`, `fromChunks`,
   `toChunks`, `empty`, `null`, `length`, `hPut`, `readFile`, `writeFile`

5. **Data.Sequence** — `singleton`, `fromList`, `empty`, `(|>)`, `(<|)`,
   `viewl`, `viewr`, `null`, `length`

6. **Data.Functor** — `$>`, `<&>`, `void`

7. **Data.Char** — `isAlpha`, `isDigit`, `isAlphaNum`, `isSpace`,
   `isUpper`, `isLower`, `toLower`, `toUpper`, `ord`, `chr`,
   `isLetter`, `isPunctuation`

**Implementation:** In `check_files_ordered()` / `check_unit_for_multimodule()`,
when a module imports a builtin, synthesize a `ModuleExports` with the known
symbol names (DefKind::Value for functions, DefKind::Type for types,
DefKind::Constructor for data constructors). The types can be left as
`Scheme::mono(Ty::var("a"))` placeholder — for `bhc check` we only need
name resolution, not full type checking of imported symbols.

**Where:** `crates/bhc-driver/src/lib.rs` — new `builtin_module_exports()`
method that returns `Option<ModuleExports>` for known builtins.

### P2: Fix cascade-critical internal modules (Category E)

Identify which internal modules are most imported and fix them:

1. **Text.Pandoc.Shared** — exports `literal`, `tshow`, `defField`,
   `stringify`, `splitTextBy`, etc. Currently fails because it imports
   from Text.DocLayout (`$$`, `blankline`, `vcat`) and Data.Aeson (`.=`).
   If we add DocLayout and Aeson to builtin stubs (P1-extended), this
   module might pass.

2. **Text.Pandoc.XML** — exports `mknode`, `inTags`, `escapeStringForXML`.
   Check what it needs.

3. **Text.Pandoc.Class** — exports `report`, `PandocMonad` class.
   Complex — depends on many things.

4. **Text.Pandoc.Options** — exports `isEnabled`, `Extension`,
   `ReaderOptions`, `WriterOptions`.

5. **Text.Pandoc.Builder** — exports `str`, `text`, `space`, `plain`,
   `para`, `divWith`, `spanWith`, etc. This is from pandoc-types
   package, so needs stub.

### P3: pandoc-types ADT stubs (Category B)

Create a stub `.hs` file (or builtin exports) for `Text.Pandoc.Definition`
that exports the core Pandoc ADTs:

```haskell
-- Types needed:
data Pandoc = Pandoc Meta [Block]
data Block = Plain [Inline] | Para [Inline] | LineBlock [[Inline]]
           | CodeBlock Attr String | RawBlock Format String
           | BlockQuote [Block] | OrderedList ListAttributes [[Block]]
           | BulletList [[Block]] | DefinitionList [([Inline], [[Block]])]
           | Header Int Attr [Inline] | HorizontalRule
           | Table Attr Caption [ColSpec] TableHead [TableBody] TableFoot
           | Figure Attr Caption [Block] | Div Attr [Block]
data Inline = Str Text | Emph [Inline] | Underline [Inline]
            | Strong [Inline] | Strikeout [Inline] | Superscript [Inline]
            | Subscript [Inline] | SmallCaps [Inline] | Quoted QuoteType [Inline]
            | Cite [Citation] [Inline] | Code Attr Text | Space | SoftBreak
            | LineBreak | Math MathType Text | RawInline Format Text
            | Link Attr [Inline] Target | Image Attr [Inline] Target
            | Note [Block] | Span Attr [Inline]
type Attr = (Text, [Text], [(Text, Text)])
nullAttr :: Attr
type Target = (Text, Text)
-- etc.
```

This could be placed in `/tmp/pandoc-smoke/src/` or built into the driver
as a "well-known package" export table.

### P4: Text.DocLayout stubs (Category C — `$$` operator)

`$$` appears 496 times. Text.DocLayout provides pretty-printing combinators.
A stub with the key exports:

```haskell
-- Key exports:
($$), ($+$), (<+>), empty, blankline, space, cr, text, char,
nest, hang, vcat, hcat, hsep, vsep, chomp, render, Doc
```

### P5: Text.Parsec stubs (Category D)

Create builtin exports for Text.Parsec modules:

- `Text.Parsec` — re-exports
- `Text.Parsec.Char` — `char`, `string`, `anyChar`, `letter`, `digit`,
  `space`, `newline`, `satisfy`, `oneOf`, `noneOf`
- `Text.Parsec.Combinator` — `many1`, `manyTill`, `lookAhead`,
  `notFollowedBy`, `skipMany`, `skipMany1`, `option`, `optionMaybe`,
  `optional`, `try`, `choice`, `count`, `between`, `sepBy`, `sepBy1`,
  `endBy`, `endBy1`, `chainl1`, `chainr1`, `eof`
- `Text.Parsec.Prim` — `getState`, `putState`, `updateState`,
  `getPosition`, `setPosition`, `getInput`, `try`, `Parsec`, `ParsecT`

### P6: Investigate type checking failures (Category F)

Debug the 7 modules that pass lowering but fail type checking.
Start with `Future` (1 error) and work up. These reveal real bugs
in the type checker.

### P7: Data.Aeson stubs

For `.=`, `.:?`, `.!=`, `object`, `pairs` etc. Lower priority since
it mostly affects Options/Shared modules.

---

## Near-passing modules (quick wins after P1)

These modules have 1-2 errors and could pass with targeted fixes:

| Module | Errors | Blocking name | Fix via |
|--------|--------|--------------|---------|
| Version | 1 | `T.pack` | P1 (Data.Text exports) |
| Namespaces | 1 | `T.isPrefixOf` | P1 (Data.Text exports) |
| TagCategories | 1 | `unions` | P1 (Data.Set/Map exports) |
| CommonState | 1 | `WARNING` ctor | P2 cascade (Logging module) |
| Utils (XML) | 2 | `T.unpack`, `lex` | P1 + base export |
| Image | 2 | `pipeProcess`, `tshow` | P2 cascade (Shared, Process) |
| Scripting | 2 | `PandocNoScriptingEngine` | P2 cascade (Error module) |
| BakedIn | 2 | `splitDirectories`, `dataFiles'` | P2 cascade |

After P1, at minimum **Version** and **Namespaces** should start passing
(total: 12). With cascade effects from other fixes, potentially 20+.

---

## Measurement

After each change, re-run:
```bash
bhc -I /tmp/pandoc-smoke/src check /tmp/pandoc-smoke/src/
```

Track: `N passed, M failed, K skipped`

| Date | Passed | Failed | Skipped | Change |
|------|--------|--------|---------|--------|
| 2026-03-01 | 10 | 195 | 16 | Baseline (multi-module check) |
