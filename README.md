<div align="center">

<a href="https://arcanist.sh/bhc/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/static/bhc-lockup-dark.svg">
    <img src=".github/static/bhc-lockup.svg" alt="bhc" height="72">
  </picture>
</a>

<h3>A Haskell compiler built to check.</h3>

<p>
  <a href="https://github.com/arcanist-sh/bhc/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/arcanist-sh/bhc?color=8839ef&label=release&style=flat-square"></a>
  <a href="https://github.com/arcanist-sh/bhc/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/arcanist-sh/bhc/ci.yml?branch=main&color=8839ef&label=ci&style=flat-square"></a>
  <a href="LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-8839ef?style=flat-square"></a>
</p>

<p>
  <a href="https://arcanist.sh/bhc/">Website</a> ·
  <a href="https://docs.arcanist.sh/bhc/">Documentation</a> ·
  <a href="https://arcanist.sh/bhc/status/">Status</a> ·
  <a href="https://arcanist.sh/bhc/playground/">Playground</a> ·
  <a href="https://arcanist.sh/bhc/blog/">Blog</a>
</p>

</div>

---

bhc, the Basel Haskell Compiler, is a clean-slate compiler for the Haskell you
already write. It is written in Rust, compiles to native code through LLVM and
to WebAssembly for WASI, and lets you choose the runtime profile a program is
built for. It shares no code with GHC.

```bash
curl -fsSL https://arcanist.sh/bhc/install.sh | sh
export PATH="$HOME/.bhc/bin:$PATH" LIBRARY_PATH="$HOME/.bhc/lib:$LIBRARY_PATH"
echo 'main = putStrLn "Hello from bhc!"' > hello.hs
bhc hello.hs -o hello && ./hello
```

Release archives are built for macOS on Apple silicon and Linux on x86_64.
With [hx](https://arcanist.sh/hx/), `hx toolchain install --bhc 0.2.22`
installs bhc next to GHC, and `hx build --backend bhc` builds a Cabal project
with it.

## Where it stands

bhc is in beta. Its yardstick is [Pandoc](https://pandoc.org): about 60,000
lines of real-world Haskell over some 80 packages, with type classes, monad
transformer stacks and Template Haskell.

| | Today |
|---|---|
| Pandoc, `bhc check` | 221 / 221 modules |
| Pandoc, compile to native | 221 / 221 modules; the binary links |
| Pandoc, as a library | `readMarkdown` and `writeHtml5String` run end to end |
| WebAssembly | 236 / 243 differential fixtures byte-identical to native, in wasmtime |
| End-to-end fixtures | 278 / 278 |
| GHC differential | 219 programs agree with GHC, 0 diverge |

The [status page](https://arcanist.sh/bhc/status/) keeps the measured details
and the caveats. The main one: the garbage collector is a tested module that
compiled programs do not use yet, so programs allocate without freeing, which
suits batch work and not long-running servers. Known bugs are tracked in the
[issues](https://github.com/arcanist-sh/bhc/issues?q=is%3Aopen+label%3Abug).

## An example

Data types, a type class, `Data.Map` and `do` notation:

```haskell
module Main where

import qualified Data.Map as Map

data Shape = Square Int | Rect Int Int
  deriving (Show, Eq)

class HasArea a where
  area :: a -> Int

instance HasArea Shape where
  area (Square s) = s * s
  area (Rect w h) = w * h

main :: IO ()
main = do
  let shapes = [("tile", Square 3), ("door", Rect 1 2)]
      byName = Map.fromList shapes
  mapM_ (\(name, s) -> putStrLn (name ++ ": " ++ show (area s))) shapes
  print (fmap area (Map.lookup "door" byName))
  print (sum (map (area . snd) shapes))
```

```console
$ bhc Main.hs -o shapes && ./shapes
tile: 9
door: 2
Just 2
11
```

## Profiles and targets

A profile is a contract between a program and the runtime it runs on. The
language stays the same; you pick the profile per build, with `--profile` or
for a whole project in `hx.toml`.

| Profile | Today |
|---|---|
| `default` | Lazy evaluation, the everyday path |
| `numeric` | Fuses `map`/`map`, `foldl'`/`map`, and `sum`/`product`/`foldl'` over `Int` ranges into single loops on native code; the Tensor IR pipeline and `--kernel-report` sit alongside |
| `server` | Compiles like `default`; the structured-concurrency scheduler and STM exist in the runtime and are not yet reachable from compiled programs |
| `edge` | A smaller runtime footprint, mainly for WebAssembly |

```bash
bhc Main.hs -o app                                    # native, through LLVM
bhc --target wasm32-wasi Main.hs -o app.wasm          # WASI module, no LLVM
bhc --profile=numeric --kernel-report Kernel.hs       # numeric profile with its kernel report
bhc check src/Main.hs                                 # type-check only
```

The WebAssembly backend emits WASI modules directly and runs them in
`wasmtime` with host file IO through a `--dir` preopen. A Metal runtime
executes kernels on Apple silicon GPUs; CUDA and ROCm code generation is
validated against mocks only.

## Tools

| Command | What it does |
|---|---|
| `bhc` | The compiler driver: `build`, `check`, `run`, `repl`, `version`, `completions` |
| `bhci` | An interactive REPL over the Core interpreter (`bhc repl` starts it) |
| `bhi` | An IR and kernel-report inspector |
| `bhc-docs` | Generates the standard library [API reference](https://docs.arcanist.sh/bhc/api/) |

Release archives ship `bhc` and its libraries; `bhci`, `bhi` and `bhc-lsp` come
from a source build for now (#50).

The [playground](https://arcanist.sh/bhc/playground/) runs bhc's front end and
interpreter, compiled to WebAssembly, in the browser.

## Repository layout

```
crates/        the compiler: lexer, parser, lowering, type checker, Core and its optimizer,
               Tensor and Loop IR, LLVM and WebAssembly backends, GPU kernels, driver, LSP
rts/           the runtime: bhc-rts (linked into every program), plus the allocator, arena,
               collector and scheduler crates
stdlib/        Rust support for base, containers, text, transformers and the rest
tools/         bhci, bhi and bhc-docs
tests/, crates/bhc-e2e-tests/
               conformance tests, end-to-end fixtures and the GHC differential harness
spec/          design documents, briefs and reviews
docs/          a short companion to the documentation site
```

How the pieces fit together is in the
[compiler architecture](https://docs.arcanist.sh/bhc/docs/internals/architecture/)
guide.

## Building from source

You need the Rust toolchain pinned in `rust-toolchain.toml` (1.96.0; `mise install`
sets it up) and LLVM 21.

```bash
cargo build -p bhc -p bhc-rts -p bhc-base -p bhc-containers -p bhc-text   # compiler and the libraries it links
cargo test --all-targets --no-fail-fast                                    # unit tests and the end-to-end suites
LIBRARY_PATH=target/debug ./target/debug/bhc hello.hs -o hello
```

Building only `bhc` leaves out the static libraries that native programs link
against, so build the runtime and library crates with it, as above. CI also
runs `crates/bhc-e2e-tests/ghc_differential.py`, which compiles every fixture
with bhc and GHC and compares the output.

## Contributing

Issues and pull requests are welcome. The coding guidelines are in
`.claude/rules/`; commits follow the conventional style (`feat:`, `fix:`,
`perf:`, `refactor:`, `docs:`, `test:`, `chore:`). A change lands when `cargo fmt`,
`cargo clippy -- -D warnings` and the full test run pass.

## Why "Basel"

bhc is built in Basel, Switzerland, and named after it: a nod to the Glasgow
Haskell Compiler, and to HBC, the Haskell B. Compiler, whose simplifier and
pattern-match compilation bhc learned from. Why a second compiler at all, and
what bhc is not, is on [bhc vs GHC](https://arcanist.sh/bhc-vs-ghc/).

## License

BSD-3-Clause. bhc is part of [arcanist.sh](https://arcanist.sh), with
[hx](https://github.com/arcanist-sh/hx). Haskell is a community language;
bhc is one implementation of it and does not define the standard.
