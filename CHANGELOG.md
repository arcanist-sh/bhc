# Changelog

All notable changes to BHC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.23] - 2026-10-22

The pandoc release: pandoc's markdown reader and HTML writer now run end to
end, compiled by BHC as a library under the closure IO model. On a 27-case
markdown matrix (paragraphs, headings, emphasis, links, autolinks, images,
code, bullet/ordered/nested lists, block quotes, rules, quotes, `&`/`<`,
entities) the output matches real pandoc byte for byte, and every pandoc
extension can be enabled without crashing.

### Added

- **Closure IO model** behind `BHC_IO_MODEL=closure`: `IO` is an ordinary
  closure, and the monad transformers (`StateT`, `ReaderT`, `ExceptT`,
  `WriterT`, `Identity`) are compiled as a Haskell library instead of being
  hard-wired into codegen (BRIEF-0005).
- **Monad witness dictionaries:** `Monad`, `Applicative`, `MonadPlus` and
  `Alternative` constraints are dictionary-passed, with builtin dictionaries
  for `Maybe` and lists, so monad-polymorphic code runs at those types.
- **Cross-module monomorphization** of polymorphic-monad functions at concrete
  transformer stacks, via `.bhc` Core-body sidecars.
- Codegen for the transformer stacks pandoc uses, such as `ReaderT` over
  `ExceptT` over `StateT`.
- `MonadFail` as a builtin class: `fail` dispatches to the monad's own
  instance.
- Builtins: `Data.Text.split`, `nubOrd`, `reads` at `Int`, `Data.Default.def`,
  `queryTerminal`, `$>`, nine more `Data.Text` functions, and a
  `Data.Sequence` backed by cons lists.
- Containers compare enum-like keys (every field-less constructor before any
  constructor with fields) by constructor tag, so `Set`/`Map` work with
  derived-`Ord` enumerations.
- A differential test harness against GHC, run in CI.
- Debugging aids: `BHC_DUMP_LLVM`, `BHC_DBG_PAP`, and the pandoc integration
  harness scripts under `tools/pandoc-harness/`.

### Fixed

- **Pattern matching:**
  - A clause whose guards all fail now falls through to the next clause in
    multi-argument functions and in local functions (`where`, `let … in`,
    `do`-block `let`).
  - A `do`-block `let` keeps all of a function's clauses.
  - View patterns and as-patterns over refutable patterns fall through, and
    keep source order.
  - Text literal patterns match when nested in tuples, lists and constructors.
- **Equality and lookup:**
  - `elem`, `notElem` and `lookup` compare by the element or key type's `Eq`
    instead of by address.
  - `==` on `Text` fields of a parametric constructor compares by value.
  - List and tuple equality use the element types.
- **Dictionaries and dispatch:**
  - Constrained values get functional-dependency-completed dictionaries.
  - A method-level dictionary is passed only for real class-method references.
  - Class methods passed as values resolve at their class parameter.
  - Methods inside instance bodies dispatch correctly.
- **Imports and interfaces:**
  - An import whose name shadows a codegen builtin wins over the builtin.
  - Constructors reached through qualified imports keep their interface tags.
  - A builtin-shadowing name stays module-qualified when its module's
    interface is converted again.
- **Laziness:** an argument that can diverge is passed as a thunk, and a
  recursive knot is passed lazily only when it calls the function being
  defined.
- **Values:**
  - `pure`/`return` used as a value at a list or `Maybe` type.
  - `intersperse` separator order.
  - Hexadecimal `digitToInt`/`intToDigit`.
  - `Double` arithmetic and printing.
  - `Integer` conversions and `(^)`.
- **Codegen:**
  - Value-position `>>=`, `>>`, `lift`, `modify`, `put` and `gets` respect the
    transformer stack.
  - Partial applications of builtins.
  - `Map Char` keys.
  - A non-BHC exception no longer crashes the uncaught-exception handler.

### Known issues

- `print` of an inline polymorphic call can print a pointer; bind the value
  first.
- Equality through a polymorphic `Eq a` constraint compares boxed values
  (such as `Text`) by address.
- Generic `many`/`some` for user `Alternative` instances, and
  `Data.Text.splitAt`, are stubs.

## [0.2.22] - 2026-09-02

Earlier releases are recorded as git tags; see the
[releases page](https://github.com/arcanist-sh/bhc/releases).

[Unreleased]: https://github.com/arcanist-sh/bhc/compare/v0.2.23...HEAD
[0.2.23]: https://github.com/arcanist-sh/bhc/compare/v0.2.22...v0.2.23
[0.2.22]: https://github.com/arcanist-sh/bhc/releases/tag/v0.2.22
