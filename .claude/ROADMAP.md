# BHC Roadmap — Boot Libraries vs Third-Party Packages

**Document ID:** BHC-ROAD-0003
**Status:** Active
**Last Updated:** 2026-03-04

---

## Where We Are

BHC parses 100% of Pandoc's 237 source files. The compiler builds cleanly
(33 crates, 0 errors) and compiles real Haskell programs to native executables
via LLVM. **190+ E2E tests** across 70 milestones. Separate compilation pipeline
complete (`.bhi` interfaces, `-c` mode, `--package-db`). The hx package manager
integration is wired.

**Pandoc check status:** 7 passed, 29 failed, 201 skipped out of 237 modules.
The 201 skipped modules fail on unresolved imports from external packages. The
29 failures are type-checking errors in modules whose imports do resolve.

**~199 modules are stubbed** in `builtin_module_set()` to satisfy Pandoc's
imports. Roughly half are GHC boot libraries that BHC should provide natively;
the other half are third-party Hackage packages that should eventually compile
from source.

This roadmap classifies every stubbed module and defines the path from here to
a fully compiled Pandoc binary.

---

## Classification Principle

| Category | Definition | BHC Strategy |
|----------|-----------|--------------|
| **GHC Boot Library** | Ships with every GHC installation. Part of the "platform" all Haskell code assumes. | BHC provides native (Rust) implementations, exposed as built-in packages. Stubs are permanent — they become real implementations. |
| **Third-Party Package** | Published on Hackage. Users install via cabal/stack. | Compile from Haskell source via `hx build` once the compilation pipeline supports it. Stubs are temporary. |

---

## Tier 1 — GHC Boot Libraries (BHC Provides Natively)

These packages ship with GHC. BHC must provide equivalent modules as built-in
packages. The stubs in `builtin_module_set()` for these modules are the
*permanent* API surface — they need correct exports and type signatures, not
removal.

### base (+ ghc-prim, ghc-bignum)

The foundational package. Every Haskell program depends on it.

| Module | Status |
|--------|--------|
| `Prelude` | 🟢 Stubbed |
| `Data.List` | 🟢 Stubbed (70+ ops) |
| `Data.Maybe` | 🟢 Stubbed |
| `Data.Either` | 🟢 Stubbed |
| `Data.Char` | 🟢 Stubbed |
| `Data.String` | 🟢 Stubbed |
| `Data.Int` | 🟢 Stubbed |
| `Data.Word` | 🟢 Stubbed |
| `Data.IORef` | 🟢 Stubbed |
| `Data.Tuple` | 🟢 Stubbed |
| `Data.Bool` | 🟢 Stubbed |
| `Data.Ord` | 🟢 Stubbed |
| `Data.Eq` | 🟢 Stubbed |
| `Data.Typeable` | 🟢 Stubbed |
| `Data.Data` | 🟢 Stubbed |
| `Data.Monoid` | 🟢 Stubbed |
| `Data.Semigroup` | 🟢 Stubbed |
| `Data.Function` | 🟢 Stubbed |
| `Data.Foldable` | 🟢 Stubbed |
| `Data.Traversable` | 🟢 Stubbed |
| `Data.Void` | 🟢 Stubbed |
| `Data.Proxy` | 🟢 Stubbed |
| `Data.Functor` | 🟢 Stubbed |
| `Data.Functor.Identity` | 🟢 Stubbed |
| `Data.Bits` | 🟢 Stubbed |
| `Data.Complex` | 🟢 Stubbed |
| `Data.Dynamic` | 🟢 Stubbed |
| `Data.Fixed` | 🟢 Stubbed |
| `Data.Ratio` | 🟢 Stubbed |
| `Data.Unique` | 🟢 Stubbed |
| `Data.Version` | 🟢 Stubbed |
| `Data.STRef` | 🟢 Stubbed |
| `Data.Coerce` | 🟢 Stubbed |
| `Data.List.NonEmpty` | 🟢 Stubbed |
| `Data.Bifunctor` | 🟢 Stubbed |
| `Control.Monad` | 🟢 Stubbed |
| `Control.Applicative` | 🟢 Stubbed |
| `Control.Exception` | 🟢 Stubbed |
| `Control.Concurrent` | 🟢 Stubbed |
| `Control.Concurrent.MVar` | 🟢 Stubbed |
| `Control.Arrow` | 🟢 Stubbed |
| `Control.Category` | 🟢 Stubbed |
| `Control.Monad.ST` | 🟢 Stubbed |
| `Control.Monad.IO.Class` | 🟢 Stubbed |
| `Control.Monad.Fail` | 🟢 Stubbed |
| `Control.Monad.Fix` | 🟢 Stubbed |
| `System.IO` | 🟢 Stubbed |
| `System.IO.Error` | 🟢 Stubbed |
| `System.IO.Unsafe` | 🟢 Stubbed |
| `System.Environment` | 🟢 Stubbed |
| `System.Exit` | 🟢 Stubbed |
| `System.Info` | 🟢 Stubbed |
| `System.Mem` | 🟢 Stubbed |
| `System.Timeout` | 🟢 Stubbed |
| `System.CPUTime` | 🟢 Stubbed |
| `System.Console.GetOpt` | 🟢 Stubbed |
| `Text.Read` | 🟢 Stubbed |
| `Text.Show` | 🟢 Stubbed |
| `Text.Printf` | 🟢 Stubbed |
| `Text.ParserCombinators.ReadP` | 🟢 Stubbed |
| `Text.ParserCombinators.ReadPrec` | 🟢 Stubbed |
| `Numeric` | 🟢 Stubbed |
| `Debug.Trace` | 🟢 Stubbed |
| `Unsafe.Coerce` | 🟢 Stubbed |
| `Foreign` | 🟢 Stubbed |
| `Foreign.Ptr` | 🟢 Stubbed |
| `Foreign.C` | 🟢 Stubbed |
| `Foreign.C.Types` | 🟢 Stubbed |
| `Foreign.C.String` | 🟢 Stubbed |
| `Foreign.Storable` | 🟢 Stubbed |
| `Foreign.Marshal` | 🟢 Stubbed |
| `Foreign.Marshal.Alloc` | 🟢 Stubbed |
| `Foreign.ForeignPtr` | 🟢 Stubbed |
| `GHC.Generics` | 🟢 Stubbed |
| `GHC.IO` | 🟢 Stubbed |
| `GHC.IO.Handle` | 🟢 Stubbed |
| `GHC.IO.Exception` | 🟢 Stubbed |
| `GHC.Base` | 🟢 Stubbed |
| `GHC.Show` | 🟢 Stubbed |
| `GHC.Read` | 🟢 Stubbed |
| `GHC.Num` | 🟢 Stubbed |
| `GHC.Real` | 🟢 Stubbed |
| `GHC.Enum` | 🟢 Stubbed |
| `GHC.Float` | 🟢 Stubbed |
| `GHC.Err` | 🟢 Stubbed |
| `GHC.List` | 🟢 Stubbed |
| `GHC.Stack` | 🟢 Stubbed |
| `GHC.Exts` | 🟢 Stubbed |
| `GHC.TypeLits` | 🟢 Stubbed |
| `GHC.TypeNats` | 🟢 Stubbed |
| `GHC.Types` | 🟢 Stubbed |
| `GHC.Prim` | 🟢 Stubbed |
| `GHC.Word` | 🟢 Stubbed |
| `GHC.Int` | 🟢 Stubbed |
| `GHC.Ptr` | 🟢 Stubbed |
| `GHC.ForeignPtr` | 🟢 Stubbed |
| `GHC.Conc` | 🟢 Stubbed |
| `GHC.STRef` | 🟢 Stubbed |
| `GHC.IORef` | 🟢 Stubbed |

**Total: ~90 modules**

### containers

| Module | Status |
|--------|--------|
| `Data.Map` | 🟢 Stubbed |
| `Data.Map.Strict` | 🟢 Stubbed |
| `Data.Map.Lazy` | 🟢 Stubbed |
| `Data.Map.Internal` | 🟢 Stubbed |
| `Data.Map.Strict.Internal` | 🟢 Stubbed |
| `Data.Set` | 🟢 Stubbed |
| `Data.Set.Internal` | 🟢 Stubbed |
| `Data.IntMap` | 🟢 Stubbed |
| `Data.IntMap.Strict` | 🟢 Stubbed |
| `Data.IntSet` | 🟢 Stubbed |
| `Data.Sequence` | 🟢 Stubbed |
| `Data.Tree` | 🟢 Stubbed |
| `Data.Containers.ListUtils` | 🟢 Stubbed |
| `Data.Graph` | 🟢 Stubbed (implicit via Data.Tree) |

**Note:** `Data.Containers.ListUtils` is technically in `containers` (since 0.6.3.1).

### text

| Module | Status |
|--------|--------|
| `Data.Text` | 🟢 Stubbed (25+ ops) |
| `Data.Text.Encoding` | 🟢 Stubbed |
| `Data.Text.IO` | 🟢 Stubbed |
| `Data.Text.Lazy` | 🟢 Stubbed |
| `Data.Text.Read` | 🟢 Stubbed |
| `Data.Text.Encoding.Error` | 🟢 Stubbed |
| `Data.Text.Lazy.Encoding` | 🟢 Stubbed |
| `Data.Text.Lazy.Builder` | 🟢 Stubbed |

### bytestring

| Module | Status |
|--------|--------|
| `Data.ByteString` | 🟢 Stubbed (24 ops) |
| `Data.ByteString.Char8` | 🟢 Stubbed |
| `Data.ByteString.Lazy` | 🟢 Stubbed |
| `Data.ByteString.Lazy.Char8` | 🟢 Stubbed |
| `Data.ByteString.Builder` | 🟢 Stubbed |

### transformers

| Module | Status |
|--------|--------|
| `Control.Monad.Trans` | 🟢 Stubbed |
| `Control.Monad.Trans.Class` | 🟢 Stubbed |
| `Control.Monad.Trans.State` | 🟢 Stubbed |
| `Control.Monad.Trans.State.Strict` | 🟢 Stubbed |
| `Control.Monad.Trans.State.Lazy` | 🟢 Stubbed |
| `Control.Monad.Trans.Reader` | 🟢 Stubbed |
| `Control.Monad.Trans.Writer` | 🟢 Stubbed |
| `Control.Monad.Trans.Writer.Strict` | 🟢 Stubbed |
| `Control.Monad.Trans.Writer.Lazy` | 🟢 Stubbed |
| `Control.Monad.Trans.Except` | 🟢 Stubbed |
| `Control.Monad.Identity` | 🟢 Stubbed |

### mtl

| Module | Status |
|--------|--------|
| `Control.Monad.State` | 🟢 Stubbed |
| `Control.Monad.State.Strict` | 🟢 Stubbed |
| `Control.Monad.State.Lazy` | 🟢 Stubbed |
| `Control.Monad.State.Class` | 🟢 Stubbed |
| `Control.Monad.Reader` | 🟢 Stubbed |
| `Control.Monad.Reader.Class` | 🟢 Stubbed |
| `Control.Monad.Writer` | 🟢 Stubbed |
| `Control.Monad.Writer.Strict` | 🟢 Stubbed |
| `Control.Monad.Writer.Lazy` | 🟢 Stubbed |
| `Control.Monad.Writer.Class` | 🟢 Stubbed |
| `Control.Monad.Except` | 🟢 Stubbed |
| `Control.Monad.RWS` | 🟢 Stubbed |
| `Control.Monad.RWS.Strict` | 🟢 Stubbed |

### parsec

GHC boot library since GHC 8.x. Pandoc uses it extensively.

| Module | Status |
|--------|--------|
| `Text.Parsec` | 🟢 Stubbed |
| `Text.Parsec.Char` | 🟢 Stubbed |
| `Text.Parsec.Combinator` | 🟢 Stubbed |
| `Text.Parsec.Prim` | 🟢 Stubbed |
| `Text.Parsec.Pos` | 🟢 Stubbed |
| `Text.Parsec.Error` | 🟢 Stubbed |
| `Text.Parsec.String` | 🟢 Stubbed |
| `Text.Parsec.Text` | 🟢 Stubbed |

### filepath

| Module | Status |
|--------|--------|
| `System.FilePath` | 🟢 Stubbed |
| `System.FilePath.Posix` | 🟢 Stubbed |
| `System.FilePath.Windows` | 🟢 Stubbed |

### directory

| Module | Status |
|--------|--------|
| `System.Directory` | 🟢 Stubbed |

### process

| Module | Status |
|--------|--------|
| `System.Process` | 🟢 Stubbed |

### time

| Module | Status |
|--------|--------|
| `Data.Time` | 🟢 Stubbed |
| `Data.Time.Clock` | 🟢 Stubbed |
| `Data.Time.Clock.POSIX` | 🟢 Stubbed |
| `Data.Time.Format` | 🟢 Stubbed |
| `Data.Time.Calendar` | 🟢 Stubbed |
| `Data.Time.LocalTime` | 🟢 Stubbed |

### deepseq

| Module | Status |
|--------|--------|
| `Control.DeepSeq` | 🟢 Stubbed |

### array

| Module | Status |
|--------|--------|
| `Data.Array` | 🟢 Stubbed |
| `Data.Array.IArray` | 🟢 Stubbed |
| `Data.Array.MArray` | 🟢 Stubbed |
| `Data.Array.ST` | 🟢 Stubbed |
| `Data.Array.IO` | 🟢 Stubbed |

### stm

| Module | Status |
|--------|--------|
| `Control.Concurrent.STM` | 🟢 Stubbed |
| `Control.Monad.STM` | 🟢 Stubbed |

### binary

| Module | Status |
|--------|--------|
| `Data.Binary.Get` | 🟡 Partial |

### pretty

| Module | Status |
|--------|--------|
| `Text.PrettyPrint` | 🟢 Stubbed |

### unix

| Module | Status |
|--------|--------|
| `System.Posix` | 🟢 Stubbed |
| `System.Posix.IO` | 🟢 Stubbed |
| `System.Posix.Terminal` | 🟢 Stubbed |

### exceptions

GHC boot library (ships with GHC since 8.x).

| Module | Status |
|--------|--------|
| `Control.Monad.Catch` | 🟢 Stubbed |

### network-uri

GHC boot library (ships with GHC platform).

| Module | Status |
|--------|--------|
| `Network.URI` | 🟢 Stubbed |

### random

| Module | Status |
|--------|--------|
| `System.Random` | 🟢 Stubbed |

**Boot library total: ~145 modules across ~18 packages**

---

## Tier 2 — Third-Party Packages (Compile from Source)

These are Hackage packages that Pandoc depends on. BHC should NOT permanently
stub these — they should be compiled from Haskell source once the compilation
pipeline supports it. Stubs exist only as a temporary bridge for type-checking.

Priority levels:
- **Critical** — imported by nearly every Pandoc module
- **High** — imported by many Pandoc modules or blocks a wide dependency tree
- **Medium** — imported by several modules
- **Low** — imported by few modules or can be deferred

### Critical Priority

| Package | Stubbed Modules | Notes |
|---------|----------------|-------|
| **pandoc-types** | `Text.Pandoc.Definition`, `Text.Pandoc.Builder`, `Text.Pandoc.Walk`, `Text.Pandoc.Generic`, `Text.Pandoc.JSON`, `Text.Pandoc.MediaBag` | Defines `Block`, `Inline`, `Meta` — used by every Pandoc module |
| **doclayout** | `Text.DocLayout` | Text layout engine for Pandoc writers |
| **Text.Pandoc.Translations** | `Text.Pandoc.Translations` | Internal Pandoc module (stubbed because imported early) |

### High Priority

| Package | Stubbed Modules | Notes |
|---------|----------------|-------|
| **aeson** | `Data.Aeson`, `Data.Aeson.Types`, `Data.Aeson.Encode.Pretty`, `Data.Aeson.TH` | JSON — used pervasively. Deps: scientific, vector, hashable |
| **vector** | `Data.Vector` | Used by aeson and many packages |
| **scientific** | `Data.Scientific` | Used by aeson for numeric values |
| **doctemplates** | `Text.DocTemplates`, `Text.DocTemplates.Internal` | Template engine for Pandoc |
| **skylighting** | `Skylighting`, `Skylighting.Types`, `Skylighting.Parser` | Syntax highlighting |
| **citeproc** | `Citeproc`, `Citeproc.Types`, `Citeproc.Pandoc`, `Citeproc.Locale`, `Citeproc.CslJson` | Citation processing |
| **texmath** | `Text.TeXMath`, `Text.TeXMath.Types`, `Text.TeXMath.Readers.OMML`, `Text.TeXMath.Readers.MathML.EntityMap`, `Text.TeXMath.Unicode.ToTeX`, `Text.TeXMath.Shared` | LaTeX math conversion |
| **commonmark** | `Commonmark`, `Commonmark.Entity`, `Commonmark.Extensions`, `Commonmark.Pandoc` | CommonMark parsing |
| **tagsoup** | `Text.HTML.TagSoup`, `Text.HTML.TagSoup.Tree`, `Text.HTML.TagSoup.Entity`, `Text.HTML.TagSoup.Match` | HTML parsing |
| **data-default** | `Data.Default`, `Data.Default.Class` | Default values — leaf dep |
| **safe** | `Safe`, `Safe.Foldable` | Safe partial functions — leaf dep |
| **syb** | `Data.Generics` | Scrap Your Boilerplate — generic programming |

### Medium Priority

| Package | Stubbed Modules | Notes |
|---------|----------------|-------|
| **blaze-html / blaze-markup** | `Text.Blaze`, `Text.Blaze.Internal`, `Text.Blaze.Html`, `Text.Blaze.Html.Renderer.Text`, `Text.Blaze.XHtml5`, `Text.Blaze.XHtml5.Attributes`, `Text.Blaze.XHtml1.Transitional`, `Text.Blaze.XHtml1.Transitional.Attributes` | HTML generation |
| **xml-light** | `Text.XML.Light`, `Text.XML.Light.Output` | XML processing |
| **pandoc-xml-light** | `Text.Pandoc.XML.Light`, `Text.Pandoc.XML.Light.Types`, `Text.Pandoc.XML.Light.Proc`, `Text.Pandoc.XML.Light.Output` | Pandoc's XML bridge |
| **http-client** | `Network.HTTP.Client`, `Network.HTTP.Client.Internal`, `Network.HTTP.Client.TLS` | HTTP requests |
| **http-types** | `Network.HTTP.Types`, `Network.HTTP.Types.Header`, `Network.HTTP.Types.Status` | HTTP type definitions |
| **network** | `Network.Socket` | Low-level networking |
| **connection / tls** | `Network.Connection`, `Network.TLS`, `Network.TLS.Extra` | TLS support |
| **zip-archive** | `Codec.Archive.Zip` | ZIP file handling |
| **zlib** | `Codec.Compression.GZip`, `Codec.Compression.Zlib` | Compression |
| **attoparsec** | `Data.Attoparsec.Text`, `Data.Attoparsec.ByteString`, `Data.Attoparsec.ByteString.Char8` | Fast parsing |
| **case-insensitive** | `Data.CaseInsensitive` | Case-insensitive text |
| **split** | `Data.List.Split` | List splitting — leaf dep |
| **text-collate** | `Text.Collate`, `Text.Collate.Lang` | Unicode collation |
| **yaml** | `Data.Yaml`, `Data.Yaml.Internal`, `Text.Libyaml` | YAML processing |

### Low Priority

| Package | Stubbed Modules | Notes |
|---------|----------------|-------|
| **JuicyPixels** | `Codec.Picture`, `Codec.Picture.Metadata` | Image processing |
| **base64-bytestring** | `Data.ByteString.Base64`, `Data.ByteString.Base64.Lazy` | Base64 encoding |
| **unicode-transforms** | `Data.Text.Normalize` | Unicode normalization |
| **file-embed** | `Data.FileEmbed` | TH-based file embedding |
| **crypton** | `Crypto.Hash` | Cryptographic hashing |
| **text-conversions** | `Data.Text.Conversions` | Text type conversions |
| **unicode-data** | `Unicode.Char` | Unicode character properties |
| **djot** | `Djot`, `Djot.AST` | Djot markup |
| **typst** | `Typst`, `Typst.Types`, `Typst.Methods` | Typst format |
| **jira-wiki-markup** | `Text.Jira.Parser`, `Text.Jira.Printer`, `Text.Jira.Markup` | Jira wiki format |
| **haddock-library** | `Documentation.Haddock.Parser`, `Documentation.Haddock.Types` | Haddock parsing |
| **gridtables** | `Text.GridTable` | Grid table parsing |
| **pretty-show** | `Text.Show.Pretty` | Pretty printing |
| **emoji** | `Text.Emoji` | Emoji support |
| **temporary** | `System.IO.Temp` | Temp file handling |
| **x509-system** | `System.X509` | X.509 certificate store |
| **Glob** | `System.FilePath.Glob` | File globbing |
| **mime-types** | `Network.Mime` | MIME type detection |
| **ipynb** | `Data.Ipynb` | Jupyter notebook format |
| **Paths_pandoc** | `Paths_pandoc` | Auto-generated by Cabal |
| **AsciiDoc** (internal) | `AsciiDoc` | AsciiDoc format |
| **Powerpoint** (internal) | `Text.Pandoc.Writers.Powerpoint.Output` | PPTX writer internals |

**Third-party total: ~54 modules across ~40+ packages**

---

## Milestone Plan

### P1 — Complete GHC Boot Library Stubs

**Goal:** All Pandoc imports from boot libraries resolve without "unbound" errors.

**Status:** 🟡 In progress

**Work:**
- Audit all boot library modules for missing exports
- Add complete export lists matching GHC's actual exports
- Ensure type signatures are correct for all exported names
- Fix polymorphic Semigroup/Monoid instances (recently landed)

**Exit criteria:**
- Every Pandoc module that imports only boot libraries passes `bhc check`
- Zero "unbound variable" errors for names exported by GHC's base/containers/text/bytestring/transformers/mtl

### P2 — Compile Leaf Third-Party Packages from Source

**Goal:** 5+ third-party packages compile from Haskell source via `hx build`.

**Status:** 🔴 Not started

**Target packages** (minimal dependencies, no TH, no C FFI):
- `data-default` / `data-default-class`
- `safe`
- `split`
- `case-insensitive`
- `base64-bytestring`

**Work:**
- End-to-end `hx build` test with a trivial Hackage package
- Expose BHC builtins as proper packages in the package DB
- Handle conditional dependencies in `.cabal` files
- Remove stubs from `builtin_module_set()` once packages compile from source

**Exit criteria:**
- `hx build` compiles 5+ Hackage packages from source to `.bhi` + `.o`
- Compiled packages are usable as dependencies by downstream packages

### P3 — Compile pandoc-types from Source

**Goal:** `pandoc-types` compiles from source, providing real `Block`/`Inline`/`Meta` types.

**Status:** 🔴 Not started

**Why this matters:** `pandoc-types` is imported by virtually every Pandoc module.
It defines the core document AST. Compiling it from source means Pandoc's type
checking uses real types, not stubs.

**Dependencies:** base, containers, text, bytestring, aeson, deepseq, syb, transformers.
This requires the aeson chain: aeson → scientific → vector → hashable → ...

**Exit criteria:**
- `bhc check` works with `pandoc-types` compiled from source (not stubbed)

### P4 — Pandoc `bhc check` 100%

**Goal:** All 237 Pandoc modules pass type checking.

**Status:** 🔴 Not started (currently 7/237)

**Work:**
- All remaining third-party packages either compiled from source or minimally stubbed
- Fix all 29 currently-failing modules (type checking errors)
- Resolve remaining 201 skipped modules (import resolution)

**Exit criteria:**
- `bhc check` on full Pandoc 3.6.4 source: 237/237 pass
- Automated CI job runs this check

### P5 — Pandoc Native Binary

**Goal:** `bhc build pandoc` produces a working native executable.

**Status:** 🔴 Not started

**Work:**
- Full codegen for all Pandoc modules (every Core IR construct used must lower to LLVM)
- Link all 237 modules + RTS + dependencies into native binary
- Functional smoke test

**Exit criteria:**
- `echo "# Hello" | ./pandoc -f markdown -t html` produces correct output
- Pandoc's own test suite pass rate >80%

---

## Compiler Blockers for Third-Party Compilation

Key language features and type system capabilities still needed before
third-party packages can compile from Haskell source:

| Blocker | Status | Impact |
|---------|--------|--------|
| Qualified type alias resolution (`T.Text` ≠ `Text` in type checking) | 🔴 | Breaks most packages using qualified imports |
| Where-clause forward references / mutual recursion | 🔴 | Common Haskell pattern |
| OverloadedStrings in pattern matching positions | 🔴 | Used by text-heavy packages |
| Tuple-pattern `let` in `do` blocks | 🔴 | Common idiom |
| `.hs-boot` files (mutual module recursion) | 🔴 | Required for packages with circular module deps |
| Complete `deriving` for all GHC derivable classes in user code | 🟡 | 9 classes done; Data, Lift missing |
| RankNTypes (rank-2 polymorphism) | 🔴 | Used in lens, some Pandoc modules |
| ExistentialQuantification | 🔴 | Used in Pandoc's PandocMonad |
| `foreign export ccall` codegen | 🔴 | `foreign import` works; export not yet |

---

## Remaining Work (Non-Pandoc)

### WASM Backend (W1)

**Status:** 🟡 70% — emitter exists but output fails wasmtime validation

- Fix WASM binary format validation
- Hello world E2E: `putStrLn "Hello"` → WASM → wasmtime
- Verify Tensor IR → Loop IR → WASM pipeline

### GPU Backend

**Status:** 🟡 80% — PTX codegen works, needs CUDA hardware testing

- End-to-end GPU testing with CUDA hardware
- AMDGCN backend testing

### Tooling (T1)

**Status:** 🟡 Partial

- REPL evaluation (currently stubbed)
- IR inspector integration testing
- LSP: go-to-definition, hover-for-type
- `-ddump-*` flags for optimization debugging

---

## Completed Work

### Compiler Pipeline

- Lexer, parser, type checker, HIR, Core IR, LLVM codegen — all working
- 190+ E2E tests across 70 milestones (E.1–E.70)
- Native executables via LLVM (hello world through GADTs + monad transformers)
- Closures, thunks, lazy evaluation, pattern matching, ADTs
- Generational GC, work-stealing scheduler, STM

### Language Features

- 30+ GHC extensions implemented
- Full typeclass system: dictionary passing, superclasses, default methods, DeriveAnyClass, GND, DerivingStrategies
- 9 stock derivable classes: Eq, Show, Ord, Enum, Bounded, Functor, Foldable, Traversable, Read + Generic stubs
- Record syntax: named fields, accessors, construction, update, wildcards, puns
- GADTs with type refinement
- Type families (open, closed, associated) + data families
- CPP preprocessing (built-in Rust preprocessor)
- Layout rule (full Haskell 2010 Section 10.3)
- Type applications, pattern synonyms, view patterns, multi-way if, lambda-case
- Standalone deriving, empty data decls, strict data, default signatures

### Core IR Optimizer

- Simplifier: constant folding, beta reduction, case-of-known-constructor, case-of-case, dead binding elimination, inlining, occurrence analysis
- Pattern match compilation: Augustsson/Sestoft decision trees, exhaustiveness and overlap checking
- Demand analysis + worker/wrapper: boolean-tree strictness, fixpoint iteration
- Dictionary specialization: direct method selection on known dictionaries

### Separate Compilation

- `-c` mode: compile to `.o` + `.bhi` without linking
- `--odir`/`--hidir`/`--package-db` flags
- `.bhi` interface generation and consumption
- TypeConverter bridge for cross-module type checking
- hx package manager integration wired (hx-bhc crate)

### Runtime System

- Structured concurrency: withScope, spawn, await, cancel, deadlines
- STM: TVar, atomically, retry, orElse, TMVar, TQueue
- Work-stealing scheduler with crossbeam deques
- Incremental GC with tri-color marking, SATB barriers
- Realtime profile: bounded GC pauses, frame arenas
- Embedded profile: no-GC static allocation

### Numeric Pipeline

- Tensor IR with shape/stride metadata and fusion passes
- Loop IR with auto-vectorization (SIMD) and parallel loops
- Hot arena allocator, pinned buffers
- GPU backend: PTX codegen, device memory, kernel launch (mock-tested)

---

## Principles

1. **Real code drives the roadmap** — Every milestone validated against Pandoc or Hackage packages
2. **Bottom-up** — Compile leaf packages first, work up the dependency tree
3. **Boot libs are permanent, third-party stubs are temporary** — Boot library stubs become real implementations; third-party stubs get replaced by source compilation
4. **One blocker at a time** — Fix the highest-impact blocker, re-test, repeat

---

## References

- [Pandoc Compilation TODO](TODO-pandoc.md) — Detailed error catalog and compiler gaps
- [BHC Specification](../CLAUDE.md) — Project overview and architecture
- [Coding Rules](rules/) — Style, testing, optimization, IR design
