# BHC Booth Demos — Zurihac 2026

Four pre-rehearsed demos for the booth. Each is designed to land in
under 60 seconds and to fail honestly if BHC can't do the thing — no
canned recordings.

Total time budget if you run all four: **about 4 minutes**.

## Prerequisites

```bash
# One-time install (macOS / Linux):
curl -fsSL https://arcanist.sh/bhc/install.sh | sh
# Or via Homebrew:
brew install arcanist-sh/tap/bhc

# Confirm:
bhc --version
bhci --version
```

If `bhc --version` works, the demos work. The native-compile demos
(01–03) link against `libbhc_rts.a` which `install.sh` puts in
`~/.bhc/lib`. Make sure `LIBRARY_PATH` is exported per the install
instructions.

## Demo 1 — Native compilation in 30 seconds (`01-hello.hs`)

The boring-is-the-point demo. Compiles real Haskell to a native
executable via LLVM.

```bash
bhc demos/01-hello.hs -o /tmp/hello
/tmp/hello
```

Talking point: this is not an interpreter. There's no GHC underneath.
LLVM optimisation, BHC's own RTS, generational GC, the whole stack.

## Demo 2 — Type classes, ADTs, recursion (`02-types.hs`)

A small binary tree with `insert`, a user-defined `Summary` class
instantiated for `Show a => Tree a`, and three folds over the
structure. Tests dictionary passing, derived `Show`, polymorphic
recursion, multi-method classes.

```bash
bhc demos/02-types.hs -o /tmp/types
/tmp/types
```

Talking point: BHC handles real Haskell idioms — pattern matching,
type-class dispatch via dictionary passing, recursion on algebraic
data types. The example is small enough to fit on one screen but
exercises the parts of the typechecker most users care about.

## Demo 3 — A Roman numeral converter (`03-roman.hs`)

A round-tripping encoder/decoder. Lists, `Maybe`, prefix matching,
guards, list comprehensions. Closer to "real code" than hello-world
or factorial.

```bash
bhc demos/03-roman.hs -o /tmp/roman
/tmp/roman
```

Talking point: a 30-line idiomatic Haskell module that exercises
`Data.List`, `Maybe`, multi-line pattern matching, and lookup-table
traversal. Everything you'd write in production. Compiles and runs
without ceremony.

## Demo 4 — The REPL (`04-repl.txt`)

An interactive session, ~20 lines. Paste them one at a time. Comments
in the file (`-- ...`) are for you, not for the audience.

```bash
bhci
# then paste lines from demos/04-repl.txt
```

Highlights worth pointing at while typing:

- `1 /= 2` — we shipped a fix for `/=` last week (it was an alias for
  `==`); the dev-log post on
  [arcanist.sh/bhc/blog/four-parser-fixes/](https://arcanist.sh/bhc/blog/four-parser-fixes/)
  covers that and three other parser bugs.
- The factorial line — recursive `let` over a function-form binding;
  the REPL handles it correctly.
- `let greet = \name -> ...` then `greet "Zurihac"` — bindings persist
  across the session.
- The triple `(sum xs, length xs, maximum xs)` shows the type as
  `(Int, Int, Int)` from the runtime value when the typechecker leaves
  the scheme polymorphic.

## Demo 5 (stretch) — `bhc check` on real Pandoc

Only if someone asks "does it compile *my* code?" and you have a
Pandoc source tree to point at.

```bash
# With pandoc-3.6.4 source unpacked under /tmp/pandoc-src
bhc -I /tmp/pandoc-src/src check /tmp/pandoc-src/src/Text/Pandoc/UUID.hs
# Or the whole tree:
bhc -I /tmp/pandoc-src/src check /tmp/pandoc-src/src/
```

The summary at the end is the live `bhc check` Pandoc number. The
[status page](https://arcanist.sh/bhc/status/) tracks where this
number sits currently (78 / 221 at the time of the Zurihac post).

## If a demo fails at the booth

The most useful response is to capture it:

```bash
# Save the failing source and the bhc diagnostic to a paste somewhere:
bhc check thing.hs 2>&1 | tee /tmp/bhc-failure.txt
```

Then point the person at the
[issue tracker](https://github.com/arcanist-sh/bhc/issues). The
Pandoc number moves fastest when real codebases expose specific gaps,
and the same is true for any code a booth visitor brings.
