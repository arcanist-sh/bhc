# Changelog

All notable changes to BHC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Derived `Show` parenthesizes a field that is a constructor application or
  a negative number, as GHC's `showsPrec 11` does: `Node Leaf 3 (Node Leaf 4
  Leaf)`, `P (-1) 2`.
- `UnboxedTuples`: `(# a, b #)` types, expressions and patterns are accepted
  and treated as ordinary tuples, `(# x #)` as `x` and `(# #)` as `()` (#36).
  `(#` and `#)` are only unboxed-tuple brackets next to whitespace, so an
  operator section like `(#)` still lexes as before.
- A parse error is a compile error (#4, #36). The parser recovered by
  dropping what it could not parse, so `this is not haskell {{{` passed
  `bhc check` and a module could compile with declarations missing;
  `BHC_LENIENT_PARSE=1` restores that behaviour. To get there, constructs the
  parser silently dropped in 19 pandoc modules are now parsed: pattern
  synonym signatures and explicitly bidirectional synonyms, pragmas at the end
  of a module, before `module` and after a doc comment, tuple sections after
  an infix expression (`(p </> fp,)`), infix definitions with a wildcard or
  literal left operand, operator variables as parameters (`(><)`), `let … in`
  as a `do` statement, `then`/`else`/case guards at the column of a `do`
  block's statements, Haddock comments inside type signatures, qualified
  class names in instance contexts, the tuple type constructor `(,)`,
  decimal/octal/control/named character escapes (`'\026'`, `'\DEL'`), and
  string gaps (`\&` left a NUL character in the string).
  `BHC_DUMP_TOKENS=1` prints the token stream, layout tokens included.
- `traverse` over a derived `Traversable` type at `Maybe` or `Either` works
  (`traverse Just (Pair 1 2)` is `Just (Pair 1 2)`; it was `Nothing` or
  `Just 0`), and `traverse`/`for` are typed `(a -> f b) -> t a -> f (t b)`
  so the result's element type is known (#32). Recursive and nested fields
  still use the generic traversal.
- `deriving (Semigroup, Monoid) via V` on a newtype takes `V`'s instance
  methods; it derived an empty instance whose methods were error stubs
  (#35). The via type needs a real instance: `Data.Monoid`'s `Sum`,
  `Product` and friends are not implemented yet.
- `bhc run` (the interpreter): `fst`/`snd` force the component they return,
  floating literals are `Double` unless typed `Float`, and floating values
  show as Haskell does (`5.0`, `1.0e-3`). A Double computation through
  `fst` failed with `expected Num, got <thunk>` (#17).
- Derived `Show` uses record syntax for record constructors:
  `P {name = "Ada", age = 36}`, not `P "Ada" 36` (#49).
- Derived `Functor` maps through recursive and nested fields: `fmap` on
  `Node (Tree a) a (Tree a)` or `Rose a [Rose a]` applied `f` to whole
  subtrees and crashed; it now recurses, and maps inside `[a]`/`Maybe a`.
- `print` of a derived-`Show` value returned by a call
  (`print (fmap (+1) (Pair 1 2))`) shows it; it printed an address (#32).
- `fmap`/`<$>` at `Either` and `<*>` at `Maybe` and `Either` work
  (`(+) <$> Just 1 <*> Just 2`, `fmap (+1) (Right 1)`); they ran the IO
  path, crashing or returning `Just 0`/`Right 0` (#23).
- `sum`, `product`, `foldl`/`foldr` and lambdas over `Integer` and `Double`
  lists compute in bignums and doubles (#18, #22). The type checker tied a
  fold's element type to nothing, so it defaulted to `Int`; `sum`/`product`
  and operators passed as values (`foldr (+)`) did machine-int arithmetic;
  `sum [1, 2, 3 :: Integer]` crashed. `[1 .. 3] :: [Double]` is a list of
  Doubles, and a `Double` inside a list shows as `1.0`, not `1`. Ranges at
  `Integer` are still built from machine ints.
- `readLn` works in native code (it had no implementation and returned an
  address): it reads a line and parses it at the type it is used at, as an
  `Int` when that type is left open.
- `show`/`print` of tuples with 3 or more components, of tuples built inside
  a lambda (`map (\x -> (x, True)) xs`), of `Char` results (`chr 66`,
  `maximum "hello"`, `mapM_ print "hi"`) and of negative numbers under a
  constructor (`Just (-3)`) match GHC; they printed numbers, addresses, or
  crashed (#24). An `Ordering` returned by a class method no longer prints
  as an empty line.
- `show` passed as a function (`map show [True, False]`) shows each element at
  its type; it showed every element as an `Int`, printing addresses.
- `show` on strings and characters escapes like Haskell, not Rust: `"it's"`
  rather than `"it\'s"`, non-ASCII as `\233` rather than `\u{e9}`, control
  characters by name (`\DEL`, `\SOH`), with `\&` where needed.
- An instance method defined as a builtin class method (`sh = show`,
  `same = (==)`) works. The builtin `Eq`/`Ord`/`Num`/`Show` instances named
  their methods by DefIds from an old builtin numbering, so their
  dictionaries held other functions (`show` was `++`), and the eta-expanded
  method's arguments had no type to dispatch the builtin on (#30).
- Kind-annotated binders in `data`, `newtype`, `class` and type family heads
  (`data Vec (n :: *) a`, `class C (f :: Type -> Type)`) are parsed; the
  declaration was dropped and its constructors and methods were unbound (#31).
- `getLine` and `getEnv` return a `String` instead of a raw C string, which
  crashed the program as soon as the result was printed (#20, #21).
  `lookupEnv` returns `Nothing`/`Just`, `getEnv` of an unset variable and
  `getLine` at end of input fail with an error instead of a null pointer.
- `read` works at `Double`, `[Int]` and `[Double]` (`read "3.25" :: Double`,
  `read "[1,2,3]" :: [Int]`); it parsed integers only and exited with
  `no parse` (#29).
- `--dump-ir=ast|hir|core|llvm|all` prints that stage to stdout, and
  `--emit llvm-ir` prints the optimized LLVM IR; both were parsed and ignored.
  `--dump-ir=tensor|loop` and unknown `--emit` formats are errors, and
  `--edition`/`-j` warn that they have no effect yet (#43).
- `bhc repl` explains how to get `bhci` when it is not installed next to `bhc`,
  instead of failing with an OS error (#50).
- Piping bhc's output into `head` no longer panics with a broken pipe.
- An import of a module nothing can provide is an error, as in GHC, when no
  import path, package database, interface directory or stdlib path is
  configured; with any of them it stays a warning (#45).
- An import of a module bhc cannot find (not on the search paths, in a
  package database, or among the modules it provides) is reported as a
  warning instead of being ignored silently, and lowering warnings now print
  even when lowering fails, so the missing module shows up next to the
  unbound-name errors it causes (#45).
- `bhc run` no longer prints the program's output twice, and `getArgs` returns
  the arguments after the file name instead of bhc's own command line (#44).
- `--target` accepts the documented short names (`wasi`, `cuda`, `rocm`,
  `riscv64`) and rejects an unknown target with an error. Before,
  `--target=wasi` and typos silently built a native binary for the host (#42).
- Idle runtime worker threads park instead of spinning on `yield_now`. A
  compiled program waiting on stdin used about 8 cores (#39).
- Methods of multi-parameter classes whose result type wraps the class's
  monad (pandoc's `HasQuoteContext st m`) are selected from their dictionary:
  from the one in scope, from an instance whose monad parameter is a variable
  (`instance Monad m => HasQuoteContext ParserState m`), and when the method
  has constraints of its own. Before, the dictionary was passed as an
  ordinary argument and the method reached codegen as a stub.
- Inside an instance whose context constrains the inner monad, `<$>` over the
  instance's own monad no longer takes `fmap` from that inner monad.
- Inside such an instance, a do-block in the instance's own monad binds with
  that monad's `>>=` rather than the inner monad's. pandoc's
  `withQuoteContext` crashed, and with it every smart-quote parse.
- A let-bound alias of a function argument keeps the argument's type, so a
  class-constrained function applied to it gets its dictionary. Two typechecker
  bugs dropped it: let generalization read the environment without the
  current substitution, and a variable bound twice to types of one shape
  (`[c]`, then `[Text]`) kept only the second, losing `c := Text`.
- A multi-clause helper in the `where` of a local function (itself in a
  `where` or `let`) is no longer dropped. pandoc-types' `placeRowSection` called
  its nested `go` as a stub whenever a table was built.
- An under-applied constructor used as a function value (`rowLen . Row "b"`)
  builds a partial application. It used to compile to a call of a
  `stub: Row not implemented` function; pandoc-types' `simpleTable` and table
  normalization hit it.
- An imported constructor keeps its own tag and arity when a `qualified`
  import brings a same-named constructor of another type (pandoc's `Block`'s
  `Table` and Definition's `Cell` next to the HTML writer's
  `qualified AnnotatedTable as Ann`). The qualified one was registered under
  the bare name too, so a table matched `Plain` or no alternative at all, and
  building the other one called a `stub: Cell`. Codegen now also knows each
  constructor as `Type::Con#arity`, the spelling hir-to-core uses for an
  ambiguous name.
- `()` has builtin `Semigroup` and `Monoid` instances. An `RWS` computation
  with a `()` writer (pandoc's `AnnotatedTable`) combined its output through a
  `<>` stub and crashed.
- `asks fst` in a reader whose environment holds a list gets its
  `MonadReader` dictionary. The function argument's single-pass type left the
  environment type open and overrode the occurrence's resolved type, and the
  resolved type spelled the list `[] Int`, which matched no instance head.
- A local function calling a `Monad m =>` helper keeps its monad when `m` is
  an argument of the type (`RS r s m a`) rather than its head; it was
  generalized and the helper ran with no dictionary.
- `x :| xs` compiles to a list cons cell under separate compilation, matching
  how bhc types `NonEmpty a` (as `[a]`). It compiled to a `:|` stub, so
  pandoc's HTML writer crashed on table cells.
- `intersperse` and `intercalate` imported from `Data.List` keep their real
  types. They were typed as fresh variables, so pandoc's `intersperse nl <$>
  mapM …` lost its element type and the `mconcat` of the result ran as a stub.
- A function above its data type's declaration gets that type's real
  constructor tags. Constructors were registered in declaration order, so an
  earlier `case` switched on made-up tags; pandoc's HTML writer crashed on
  every table (`TablePart` is declared below `tablePartToHtml`).
- An explicit `;` separates declarations inside a layout block (`let x = 1;
  y = 2 in …`, a class or instance body). It used to end the enclosing
  declaration, so later bindings lost the function's arguments, and an
  instance written this way inside a lambda was dropped without an error.
- `mapM`, `forM`, `sequence`, `sequence_`, `mapM_` and `forM_` brought in by
  `import Control.Monad` keep their types (the Prelude spellings' schemes), so
  a point-free `fmap sequence . mapM f` runs each traversal in its own monad.
  The inner `sequence` was expanded at the enclosing monad, and pandoc's pipe
  tables lost their rows.
- `Control.Arrow`'s `(***)` and `(&&&)` work at the function arrow (closure IO
  model). Both ran as stubs; pandoc-types' `definitionList` uses `(***)`.
- A `Text` string literal matched against a constructor field inside a local
  function without a signature compares as `Text`. The field binder was
  untyped, so the literal was compared as a list and never matched; pandoc
  never recognized task lists.
- `pi` is in scope (#19).
- `Data.Map.Strict` and `Data.Map.Lazy` (and the `Data.IntMap` variants) reach
  the same container primops as `Data.Map`; their functions ran as "not
  implemented" stubs (#26).
- `Data.List.union` drops the second list's duplicates and elements already
  in the first; it was a plain append (#27).
- With all pandoc extensions enabled, emphasis, links, smart quotes, fenced
  divs, pipe tables, definition lists and task lists render (16 of 21
  extension test inputs now match pandoc).

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
