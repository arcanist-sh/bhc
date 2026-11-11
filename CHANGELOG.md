# Changelog

All notable changes to BHC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `Numeric.showFFloat`, `showHex` and `showOct`, which aborted as stubs.
  `showFFloat` rounds the shortest decimal digits as GHC does (half to even
  when exactly half).
- `Text.Printf.printf` with a literal format string: `%d %s %c %f %x %%`,
  precision, width and the `-`/`0` flags, as an `IO` action or a `String`.
  A computed format, `%e` and `%g` are still unimplemented.
- Release archives include `bhci`, the REPL that `bhc repl` starts, and `bhi`,
  the IR inspector; the install script and the Homebrew formula install them
  (#50).
- The release workflow updates the Homebrew formula in `arcanist-sh/homebrew-tap`
  (`scripts/homebrew-formula.sh`), which installs every `libbhc_*.a` instead of
  only the RTS (#47). It needs a `HOMEBREW_TAP_TOKEN` secret, and prints the
  formula to apply by hand without one.

### Added

- `unsafePerformIO`, `unsafeDupablePerformIO`, `Debug.Trace.trace` and
  `traceIO` run; they were stubs that aborted. `trace` writes its message to
  stderr before its value is evaluated.
- `traceShow`, `traceShowId`, `traceM` and `traceShowM` run too, defined as
  in base through `trace` and `show`, so a user type's derived `Show` works.
- An E2E fixture may give an `expected_stderr.txt`, checked like
  `expected.txt`.

### Fixed

- Negating a `Double` held in a variable (`negate x`, `-x`), and `signum` of
  one, did integer arithmetic on the double's bits: `negate 1.5` was `-3.0`.
  `abs` had the same fault for a lambda parameter.
- `fromIntegral`, `toInteger`, `(/)`, `negate`, `abs` and `signum` passed as
  function values at `Double` (`map fromIntegral xs`, `foldr (/) 1 xs`,
  `map negate xs`) did not convert or used integer arithmetic (#100, native
  only; the WebAssembly backend still rejects `fromIntegral` as a value).
- `read`, `readMaybe` and `reads` at a data type that derives `Read`
  (enumerations, constructors with fields, records, recursive and
  parameterized types, newtypes), at tuples, nested lists, `Maybe`, `Either`,
  `Integer`, `Bool`, `Ordering` and `()` aborted with `no parse` or read the
  wrong thing; they parse what a derived `Read` accepts (#105, native only,
  for types declared in the module being compiled).
- `Data.Map` and `Data.Set` operations whose key is a lambda parameter
  (`foldl' (\m w -> M.insertWith (+) w 1 m) M.empty ws`) compared `String`,
  `Text` and enumeration keys by address, so equal keys were never merged or
  found (#101).
- Derived `Eq` and `Ord` on a recursive parameterized type (`data Tree a =
  Leaf | Node (Tree a) a (Tree a)`) compared the subtrees by address, so two
  equal trees built separately were unequal (#104).
- `Data.Foldable`'s `for_`, `traverse_`, `foldl'` and `foldMap`, imported by
  name, aborted as unimplemented stubs, on lists too.
- `foldl`, `foldl'` and `foldMap` over a type with a derived `Foldable`, and
  `mapM_ f =<< m` at one, walked the value as a list and crashed (#104).
- `data T :: Type -> Type where ...`, a GADT declaration with a kind signature
  in place of named parameters, is parsed (#107).
- Ranges over a derived `Enum`: `[x ..]`, `[x, y ..]` and `[x, y .. z]`
  step through the constructors (they were empty, one element, or a
  crash).
- `minBound`/`maxBound` at `Int`, the sized `Int`s and `Char` are their
  real bounds; with one user `Bounded` type in the program,
  `maxBound :: Int` was that type's last constructor.
- `round` rounds half to even, as Haskell does (`round 2.5` is 2).

- `import Data.Maybe` / `import Data.Either` keep the types of `lefts`,
  `rights`, `partitionEithers`, `catMaybes`, `mapMaybe` and the rest;
  `print (lefts es)` printed addresses.
- `Data.Map.foldrWithKey`, `foldlWithKey`, `mapWithKey` and
  `filterWithKey` have their real types; their results printed as
  addresses.
- `replicateM` works in any monad: in the list monad it is every
  combination (`replicateM 2 "ab"` is `["aa","ab","ba","bb"]`); it was
  typed at `IO` and printed nothing.

- `nub`, `group`, `elem`, `notElem`, `delete`, `union` and `intersect` at
  strings and at types with a derived `Eq` compare elements by content;
  they compared addresses (`nub ["a", "b", "a"]` kept both `"a"`s).
- A derived `Eq` compares a field of a ground type by that type:
  `P 1 "a" == P 1 "a"` compared the strings' addresses and was `False`.
- `xs \\ ys` removes the first equal element for each element of `ys`, as
  `Data.List` does; it removed every occurrence.

- `import Data.List` keeps the real types of `sort`, `sortBy`, `sortOn`,
  `nub`, `group`, `isPrefixOf` and the other list functions it re-exports;
  typed as placeholders, `print (sort xs)` at a user type printed
  addresses.
- `sort`, `maximum` and `minimum` at a type with a derived `Ord` use its
  `compare`; they compared boxed values by address.
- `comparing f` is `\a b -> compare (f a) (f b)` and `on op f` is
  `\a b -> op (f a) (f b)`: `sortBy (comparing area)` with a `Double` key
  was a type error, and ``groupBy ((==) `on` isDigit)`` crashed.
- A `Map`/`Set` built from `String` keys computed at run time
  (`M.fromListWith (+) [(w, 1) | w <- words s]`) merges equal keys.
- An unannotated fractional value defaults to `Double`, as in Haskell, and
  `show` of a `Float` narrows its value: `show 3.0` printed `"0"`, and
  `show (sqrt 2)` and `show pi` printed garbage.

- Prefix negation binds as Haskell's precedence 6: `-7 `mod` 3` is `-1`
  and `-2 ^ 2` is `-4`; the minus applied to the first term only.

- WebAssembly: `fromInteger` and `toInteger` are the identity (an `Integer`
  is an `Int` there) instead of an unknown function giving 0, and
  `fromIntegral` resolved at `Int -> Double` boxes the converted value, so a
  let-bound `x = fromIntegral k` prints `10.0` rather than `0.0`.

- `fromIntegral` from a machine integer to `Double`/`Float` converts the
  value itself. It relied on a float operator seeing the call, so bound
  first (`where x = fromIntegral k`, then `x / 4`), the integer's bits were
  read as a double and printed `1.0e-323`.

- `reads` at an enumeration with derived `Read` parses a constructor name
  and returns the rest of the input: `reads "Green rest" :: [(Color,
  String)]` is `[(Green," rest")]`. It read an `Int` and returned `[]`, so
  pandoc's `readExtension` (`reads ("Ext_" ++ name)`) found nothing.

- A local binding without a signature that converts a number,
  `where w = fromInteger n` used as an `Int`, takes the type its use gives
  it (the monomorphism restriction); the conversion compiled as the
  identity and `w + 1` added 1 to the `Integer`'s address.

- `import M (x)` warns when `M` does not export `x`
  (``warning: module `M` does not export `x` ``); the name was dropped
  silently. GHC rejects this. It stays a warning because some interfaces
  bhc provides are not complete.

- A top-level infix definition whose left operand is a constructor pattern,
  `TagOpen n as ~== TagOpen p ps = …` (tagsoup's `~==`), parses; it was
  "unexpected constructor, expected declaration". Under `BHC_IO_MODEL=eager`,
  `==` on two pattern-bound `String`s in a function with a guarded clause
  compared addresses; it is now typed from its operands.

- `mask $ \restore -> …` and `uninterruptibleMask` under
  `BHC_IO_MODEL=eager`: the body was called with its arguments swapped, so
  it ran twice and `restore` crashed. It now runs once and `restore` returns
  its action's result. The eager model still runs an IO action passed to a
  function at the call, so the action given to `restore` runs masked.

- WebAssembly: `show` of a `Maybe` that a call returns renders its payload by
  type, so `lookup 2 [(1, "a"), (2, "b")]` shows `Just "b"` rather than an
  address, and a negative payload is parenthesised. `fmap` over such a call
  takes the `Maybe` arm instead of yielding `Nothing`.

- `newMVar`, `putMVar`, `tryPutMVar`, `swapMVar` and `writeChan` no longer
  evaluate the value they store, as in GHC: `newMVar (error "x")` crashed at
  once. A stored computation runs when `takeMVar`, `readMVar`,
  `tryTakeMVar` or `readChan` first reads it, in the reading thread.

- A top-level list defined in terms of itself works: `ones = 1 : ones`
  segfaulted, `nats = 0 : map (+ 1) nats` overflowed the stack, and
  `fibs = 0 : 1 : zipWith (+) fibs (tail fibs)` crashed. Inside such a
  definition, a constructor field that uses it is built lazily, a list
  builtin defers a tail that is still being evaluated instead of reporting
  `<<loop>>`, and a reference made once the list exists reads the shared
  list, so `fibs !! n` takes linear time.

- A top-level constant (CAF) is computed once and shared by every use, as in
  GHC; it was recomputed at each use. A global `IORef` made with
  `unsafePerformIO (newIORef …)` is now one `IORef`. Constants whose type is
  or holds an IO action, or is unknown, are left as they were, and a CAF that
  throws throws again at its next use. `BHC_NO_CAF_MEMO=1` turns sharing off.

- Threads, `MVar`s and `Chan`s run under the eager IO model too
  (`BHC_IO_MODEL=eager`), with preemption, `throwTo` and `killThread`; they
  compiled to stubs that aborted. Under that model `>>=`, `>>` and `forever`
  now stop at an exception, thrown or thrown to the thread, so a throw also
  ends `forever`. `mask` with `restore` still does not run there (#91), and
  WebAssembly has no threads yet; the compile-time warning now names only that
  target (#75).

- Under the eager IO model, a recursive `Monad m =>` function called at IO,
  such as `cnt g n = cnt g (n - 1) >>= g`, gives its continuation the result
  instead of a thunk's address. The recursive call was delayed because it
  calls the function being defined; a call whose result type is `m a` (or a
  bare type variable, which #64 covered only where Core kept the type) is no
  longer delayed under that model, since a thunk cannot stand in for an
  action's raw result there (#65).

- On WebAssembly, a `Monad m =>` function called at IO runs: the backend
  knows the IO dictionary's methods (`>>=`, `>>`, `fmap`, `<*>`, `pure`) as
  values, tuple constructors up to 15 fields (dictionaries are tuples), and
  shows a value typed `Maybe Int`. Before, the dictionary held zeros and the
  program trapped (#65).

- A thread in a long pure computation no longer keeps the other threads from
  running: compiled code checks for preemption on function entry and at
  allocations, where a timer asks for it once per 20 ms slice, so recursion
  and loops that reach no IO step give way too. `killThread` reaches such a
  thread at its next IO step. A thread that needs a thunk another thread is
  evaluating waits for it instead of reporting `<<loop>>` (#75).

- `SHA256SUMS.txt` lists bare archive names, so `shasum -a 256 -c
  --ignore-missing SHA256SUMS.txt` works next to a download, and the install
  script verifies the archive against it before unpacking (#46).

- `mask` defers an exception thrown to a thread with `throwTo` or
  `killThread` until the thread unmasks or blocks, and `uninterruptibleMask`
  until it unmasks, as in GHC. `throwTo` waits until the exception is
  delivered, and a forked thread starts in its parent's masking state (#75).

- `mask $ \restore -> body` runs its body, `restore` runs its action, and an
  exception thrown inside `restore` reaches the caller. Under the default IO
  model the body never ran, so `forkFinally` and anything else built on
  `mask` did not work (#91).

## [0.2.29] - 2026-11-08

Threads take turns fairly, reading stdin no longer stops other threads, deep
evaluation no longer crashes on a small stack, and four wrong results from
polymorphic code are fixed.

### Fixed

- A thread that never blocks no longer starves the others: a thread is
  preempted at an IO step once it has run for 20 ms, and an exception thrown
  to it reaches it there. Pure code is still not preempted (#75).
- Reading stdin lets other threads run while it waits: `getLine`, `getChar`,
  `getContents`, `isEOF` and their `h` forms, `Data.Text.IO.getLine`,
  `getContents`, `hGetLine` and `hGetContents`, and lazy
  `ByteString.hGetContents` (#75).
- A long chain of unevaluated thunks no longer crashes the program: `main`
  runs on a thread with a 1 GiB stack (reserved, not committed). A program
  that still runs out of stack prints `prog: stack overflow` and exits with
  status 2, as in GHC, instead of a bare segfault (#82).
- A polymorphic recursive fold such as `foldT f z (Node l x r) = foldT f (f x
  (foldT f z r)) l` returns the right value at `Int` and `Double`. The inner
  call was passed to `f` as a thunk, which was then read as a number, so
  `foldT (+) 0 tree` printed an address (#64).
- `foldr` called directly on a type with only a derived `Foldable` instance
  works. Held in a variable, the container was folded as a list, which forced
  its raw elements and crashed, and an `Int` result printed as an address
  (#87).
- `print` of a `String` that a polymorphic function returns (`print (pick True
  "yes" "no")`) prints the string. It printed an address (#86).

### Known issues

- Pure code is not preempted, and `mask` does not defer an exception thrown
  to another thread (#75).
- With `BHC_IO_MODEL=eager`, a recursive `Monad m =>` function at `IO` prints
  an address (#65).

## [0.2.28] - 2026-11-07

`throwTo` works, `killThread` runs the killed thread's handlers, and `quot`,
`rem`, `quotRem`, `divMod`, `gcd` and `lcm` work at `Integer`.

### Added

- `throwTo`, which aborted at run time, raises its exception in the other
  thread (#75).

### Fixed

- `killThread` runs the killed thread's `finally`, `bracket` and `catch`
  handlers. It used to drop the thread without them. An uncaught
  `ThreadKilled` ends a forked thread without a message, and the main thread
  with status 1, as in GHC (#75).
- `quot`, `rem`, `quotRem`, `divMod`, `gcd` and `lcm` work at `Integer`. Their types
  were pinned to `Int`, so a negated literal operand under `quot` or `rem` crashed
  `print`, `gcd` and `lcm` on `Integer`s crashed, and `quotRem`/`divMod` on
  `Integer`s crashed (#78).

### Known issues

- A polymorphic recursive fold that returns an accumulator used at an unboxed
  type (`foldT (+) 0 tree`) prints an address (#64).
- With `BHC_IO_MODEL=eager`, a recursive `Monad m =>` function at `IO` prints an
  address (#65).
- Threads take turns, so one that never blocks starves the others, and a blocking
  read such as `getLine` stops every thread. `mask` does not defer an exception
  thrown to a thread (#75).

## [0.2.27] - 2026-11-06

A correctness release. `div` and `mod` on negative `Int`s gave wrong answers, and
division by zero gave a garbage number; both are fixed, and runtime errors in pure
code are now exceptions that `try` and `catch` can see.

### Fixed

- `div` and `mod` round toward negative infinity. They truncated, so
  ``(-7) `div` 2`` was -3 and ``(-7) `mod` 2`` was -1. At `Integer`, `quot` and `rem`
  truncate, where they rounded down.
- Division by zero throws `divide by zero`, as an `ArithException` that `try` and
  `catch` can see and `show` can print. At `Int` it gave a garbage number and
  carried on.
- `head`, `tail`, `last`, `init`, `!!`, `fromJust` and `undefined` throw an exception
  that `try` and `catch` can see, where they aborted the program (#76).
- `x % y` at `Integer` reads its operands, where it read their addresses, and a
  zero denominator throws `Ratio has zero denominator`.
- The `ErrorCall` constructor works, where it aborted as a stub.

### Known issues

- A negated literal under `quot` or `rem` in a tuple annotated `Integer` crashes
  `print` (#78).
- A polymorphic recursive fold that returns an accumulator used at an unboxed
  type (`foldT (+) 0 tree`) prints an address (#64).
- With `BHC_IO_MODEL=eager`, a recursive `Monad m =>` function at `IO` prints an
  address (#65).
- Threads take turns, so one that never blocks starves the others; `killThread`
  does not run handlers, and `throwTo` still aborts (#75).

## [0.2.26] - 2026-11-05

Native code now uses the closure IO model by default, so monad transformers are
ordinary Haskell and native output matches GHC across the differential suite.
`Control.Concurrent` runs for the first time, and `try` and exception `show` now
work.

### Added

- `Control.Concurrent` runs on native code: `forkIO`, `killThread`,
  `threadDelay`, `myThreadId`, `MVar` and `Chan` operations. Threads take turns:
  one runs until it blocks, sleeps or ends, so a thread that never blocks starves
  the others. When every thread is blocked, the program reports `thread blocked
  indefinitely in an MVar operation` and exits with status 1. `killThread` does
  not run the killed thread's `finally` handlers, and `throwTo` still aborts. The
  eager IO model and WebAssembly still warn at compile time and abort at run
  time (#38).

### Changed

- The closure IO model is now the default for native code: `IO` is an ordinary
  action closure and monad transformers are the Haskell library in
  `stdlib/bhc-transformers`, embedded in the compiler so an installed `bhc`
  finds them. Against GHC on the differential suite, all 385 comparable
  fixtures now agree, with no known failures, and transformer stacks such as
  `lift get` through `ReaderT` and `runExcept` work (#37). IO-heavy loops run slower (one
  measured loop went from 3.7 s to 6.6 s); pure and list code does not change.
  WebAssembly stays on the eager model, and `BHC_IO_MODEL=eager` selects the old
  path on native. The eager transformer code will be removed in a later release.
- A program whose `main` is plainly not an `IO` action (`main = 42`) is a compile
  error, where the closure model would crash at run time. The interpreter and
  playground still evaluate such a `main` and show its value.

### Fixed

- `try` catches. It was the identity, so the exception stayed pending and the
  result was `Right`; it now returns `Left` for the exception type it is used at,
  when that type is known (#70).
- `show` and `print` of an `IOException`, `ErrorCall` or `SomeException` print its
  text, where they printed an address, and `userError` shows as `user error (…)`.
  An uncaught exception reports `program: message` (#72).
- `evaluate`, `bracket_`, `>=>`, `<=<` and `<$` at builtin functors work, where they
  aborted with `stub: … not implemented`. `>=>` and `<=<` also have their fixity,
  `infixr 1` (#71).
- The results of `zipWithM` and `filterM` print as lists, where they printed an
  address (#73), and `print ()` prints `()`, where it printed an empty line.
- `forever` runs its body under the closure IO model, and a thrown exception ends
  it.
- Under the closure IO model, a thrown exception stops the rest of a `do` block:
  an action after the throw, such as `writeIORef`, no longer runs.
- `bhc check` finds the transformer library modules a program imports.
- WebAssembly: `IO` functions used as values run. `m >>= k` with a
  continuation that is not a lambda (`>>= print`) applies it; `print`,
  `putStrLn`, `putStr`, `return` and `pure` can be passed as functions; and
  an inlined function parameter bound to a top-level function is called.
  A traversal over a derived `Traversable` is correct under `--target=wasi`
  again (#68).

### Known issues

- Runtime errors in pure code (`div` by zero, `head []`) abort instead of throwing
  an exception that `try` or `catch` can see (#76).
- A polymorphic recursive fold that returns an accumulator used at an unboxed
  type (`foldT (+) 0 tree`) prints an address (#64).
- With `BHC_IO_MODEL=eager`, a recursive `Monad m =>` function at `IO` prints an
  address (#65).
- Threads take turns, so one that never blocks starves the others; `killThread`
  does not run handlers, and `throwTo` still aborts (#75).

## [0.2.25] - 2026-11-03

Type classes over containers and numbers: `Foldable`/`Traversable` for user
types (derived and hand-written), the list `Applicative`, traversals at a
monad that is a type variable, and class-polymorphic functions at `Integer`,
`Double` and `String`, including ones imported from another module.

### Added

- Class-polymorphic functions (`sq :: Num a => a -> a`, `Show a =>`, `Ord a
  =>`, …) get a specialized copy at each representation-sensitive type they
  are used at (`Integer`, `Double`, `Float`, `Rational`, `Text`, `String`,
  lists): top-level, `let`/`where`-bound, and imported from another module at
  the numeric types. They used machine-int arithmetic at every type (#57).
- Hand-written `Traversable` instances: `traverse` is checked at
  `Applicative f => (a -> f b) -> T a -> f (T b)` and every traversal of the
  type runs the instance, at `Maybe`, `IO`, lists and a variable monad.
  Closure IO model (`BHC_IO_MODEL=closure`) only (#66).
- Using an unimplemented concurrency primitive (`forkIO`, `MVar`, `Chan`
  operations) is a compile-time warning; they still abort at run time (#38).

### Fixed

- `Foldable` for user types: `sum`, `product`, `length`, `null`, `maximum`,
  `minimum`, `elem`, `any`, `all`, `toList`, `mapM_` and `forM_` work over
  derived, hand-written and newtype `Foldable` instances and `Maybe`; derived
  `foldr` recurses into nested fields and folds through a newtype's
  container field; a hand-written recursive `foldr` dispatches to its
  instance instead of walking the value as a list (#63, #66).
- Traversals: derived `traverse` over recursive and nested fields at any
  applicative (#53); `traverse`/`mapM`/`for` inside a `Monad m =>` function
  take `>>=`/`pure` from the dictionary instead of segfaulting (#63);
  `mapM_` runs every action (#63).
- The list `Applicative`: `<*>`, `liftA2`/`liftM2` at `Maybe`, `Either` and
  lists, `map` over functions of several arguments, and `for` at `IO` (#63).
  A recursive `Applicative f =>` function at lists no longer segfaults (#66).
- `Integer`: `let` in a list comprehension, operator chains under an
  annotation (`2 ^ 64 + 5 :: Integer`) and literals beyond 64 bits (#57).
- The closure IO model generalizes local functions over `Num`/`Eq`/`Ord`/
  `Show` variables, and runs a thunked action passed to a dictionary's `>>=`
  (#57, #63).
- `Data.Monoid`/`Data.Semigroup`: names from the bundled modules shadow the
  builtin stubs, and `Endo` is real (#60).
- WebAssembly: `getArgs`/`getProgName` read WASI argv, an unsupported
  `foreign import` is a compile error instead of returning 0, and `[String]`
  shows quoted (#41).

### Known issues

- A polymorphic recursive fold that returns an accumulator used at an
  unboxed type (`foldT (+) 0 tree`) prints an address (#64); in the default
  IO model, a recursive `Monad m =>` function at `IO` does too (#65).
- WebAssembly: an `IO` function passed as an argument does not run, so a
  traversal over a derived `Traversable` is wrong under `--target=wasi`
  (#68).

## [0.2.24] - 2026-10-31

A correctness release: around fifty issue fixes in laziness, `Integer`,
containers, derived `Show`, the type checker, `bhc run` and the WebAssembly
backend, each with an end-to-end fixture.

### Added

- `Data.Monoid` and `Data.Semigroup` are real modules, bundled with the
  compiler as Haskell source and compiled with the program: `Sum`, `Product`,
  `All`, `Any`, `Dual`, `Min`, `Max` and both modules' `First`/`Last`, with
  their `Semigroup`/`Monoid` instances (#52). `DerivingVia` through them works
  (#35). `Endo` is still a stub (#60).

### Changed

- `bhc run FILE [ARGS...]` compiles the program natively to a temporary
  executable, runs it with `ARGS` and exits with its exit code, instead of
  evaluating it in the Core interpreter (#44).

### Fixed

- Laziness: corecursive lists and `iterate`/`repeat`/`cycle` are lazy (#28);
  `map`, `filter`, `concatMap`, `zipWith` and `zip` work over infinite lists
  (#54); `[n ..]` and `[a, b ..]` at `Int` are lazy instead of capped at 10000
  elements (#58).
- `Integer`:
  - ranges build bignums, and `Integer` elements show, compare and test
    parity by value (#51);
  - literals beyond 64 bits parse;
  - literal patterns match by value;
  - `negate`/`abs`/`signum` work at `Integer`;
  - an annotation fixes the type of a whole operator chain
    (`2 ^ 64 + 5 :: Integer`), and a list-comprehension `let` takes its
    generator's type (#57).
- Containers:
  - `Data.Map`/`Data.Set` order boxed keys by content, and a stored `0` is
    found (#25);
  - `show`/`print` of a `Map` or `Set` renders `fromList [...]` (#55);
  - `Data.IntMap`/`Data.IntSet` builtins have real types (#56).
- Derived `Show`:
  - fields whose type is a type parameter show at their types: `D True`, not
    an address (#59);
  - newtypes use record syntax;
  - a type imported from another module shows with its derived instance;
  - `deriving Show via T` shows the value as a `T` (#35).
- Type checker:
  - GADTs indexed at several types (#33);
  - associated type families at a literal (#34);
  - a user class method named like a builtin one (`empty`) (#34);
  - instance type variables no longer leak between instances, which made
    `Dual`'s `<>` loop (#60);
  - defaulting never overwrites an already-known type;
  - an integer literal pattern at `Double` matches as a float.
- `zip3`/`zipWith3`/`unzip3` have their real types.
- `fmap`/`<$>` passed as a value at `Maybe`/`Either` (#25).
- WebAssembly:
  - `appendFile` reaches the host file;
  - computed `writeFile` contents are written whole;
  - `++` keeps a literal operand beside a computed string (#40).

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

### Known issues

- A polymorphic `Num a =>` function used at `Integer` does machine-int
  arithmetic (#57), and derived `traverse` over recursive fields or other
  applicatives is wrong (#53). Both need dictionaries for the builtin classes.
- `Data.Monoid.Endo` is a stub (#60).
- `forkIO`/`MVar` abort at run time (#38); `foreign import` and `getArgs` on
  WebAssembly (#41).

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

[Unreleased]: https://github.com/arcanist-sh/bhc/compare/v0.2.29...HEAD
[0.2.29]: https://github.com/arcanist-sh/bhc/compare/v0.2.28...v0.2.29
[0.2.28]: https://github.com/arcanist-sh/bhc/compare/v0.2.27...v0.2.28
[0.2.27]: https://github.com/arcanist-sh/bhc/compare/v0.2.26...v0.2.27
[0.2.26]: https://github.com/arcanist-sh/bhc/compare/v0.2.25...v0.2.26
[0.2.25]: https://github.com/arcanist-sh/bhc/compare/v0.2.24...v0.2.25
[0.2.24]: https://github.com/arcanist-sh/bhc/compare/v0.2.23...v0.2.24
[0.2.23]: https://github.com/arcanist-sh/bhc/compare/v0.2.22...v0.2.23
[0.2.22]: https://github.com/arcanist-sh/bhc/releases/tag/v0.2.22
