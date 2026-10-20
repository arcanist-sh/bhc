# Profiles

A profile tells bhc which optimisation pipeline and runtime configuration to use
for a build. This page describes what each profile does in the current compiler.
The longer-term design is in the specification under `spec/`; this page sticks
to observable behaviour.

## Selecting a profile

The profile is chosen once per build:

```bash
bhc --profile numeric Main.hs -o main
```

or in `hx.toml`:

```toml
[compiler]
backend = "bhc"

[compiler.bhc]
profile = "numeric"
```

There is no per-module profile pragma, so every module in a build uses the
same profile.

## Overview

| Profile | Selectable | What it changes today |
|---------|------------|-----------------------|
| `default` | yes | Core simplifier, demand analysis and worker/wrapper |
| `server` | yes | Same compiled pipeline as `default` |
| `numeric` | yes | Core list-fusion rewrites; Tensor IR pipeline and kernel report |
| `edge` | yes | Skips demand analysis and worker/wrapper; smaller WASM memory |

Compiled code is lazy under every profile: infinite lists and other lazy
idioms behave the same whichever profile you pick. `realtime` and `embedded`
exist in the session and specification but cannot be selected from the
command line.

Memory: compiled programs allocate without freeing. The garbage collector
exists as a runtime module but is not connected to compiled code, which is fine
for short-lived programs such as compilers and document converters and is the
main limit for long-running ones.

## default

General Haskell. After the Core simplifier, bhc runs demand analysis and the
worker/wrapper transformation, then simplifies again.

```haskell
main :: IO ()
main = print (takeWhile (< 10) [1..])
-- [1,2,3,4,5,6,7,8,9]
```

## server

Compiles exactly like `default` today. The Rust runtime contains a
work-stealing scheduler, STM, cancellation and deadlines, but compiled programs
cannot reach them yet, and `forkIO`/`MVar` compile but abort at run time
([#38](https://github.com/arcanist-sh/bhc/issues/38)). Use `default` and
sequential code for now.

## numeric

On native builds the numeric profile adds Core-to-Core list-fusion rewrites
that run before code generation:

- `sum`, `product` and `foldl'` over an `Int` range such as `[1 .. n]`,
  optionally through a `map` of a named `Int -> Int` function, become a single
  counting loop with no intermediate list;
- `sum (map f xs)` becomes a strict `foldl'` when `f` is a named function of
  type `Int -> Int` (a lambda passed to `map` is not rewritten yet);
- `foldl' op z (map f xs)` becomes a single `foldl'`;
- `map f (map g xs)` becomes one `map`.

```haskell
module Main where

import Data.List (foldl')

double :: Int -> Int
double x = 2 * x

main :: IO ()
main = do
  print (sum (map double [1 .. 1000000]))
  print (foldl' (+) 0 (map double (map double [1 .. 1000])))
```

```bash
bhc --profile numeric SumSq.hs -o sumsq && ./sumsq
# 1000001000000
# 2002000
```

The numeric profile also runs the Tensor IR and Loop IR pipeline, and
`--kernel-report` prints its fusion, vectorisation and parallelisation report.
Native code generation does not consume the Tensor IR, so for list programs
like the one above the report shows zero tensor operations; the list rewrites
above are what change the native code. The rewrites remove intermediate lists;
values are still boxed, so they do not by themselves make numeric loops as fast
as unboxed code.

`--tensor-fusion` turns the Tensor IR pipeline on for any profile.

## edge

Intended for small WebAssembly and serverless builds. It skips demand analysis
and worker/wrapper, and on the WebAssembly target it uses a smaller linear
memory (256 KB initial, 4 MB maximum, no debug names) where the default
profile allows 16 MB and `numeric` 32 MB. Because memory is not reclaimed,
allocation-heavy programs reach the 4 MB limit sooner under `edge`.

The WebAssembly target itself is chosen with `--target wasm32-wasi` and works
with every profile:

```bash
bhc --profile edge --target wasm32-wasi Main.hs -o main.wasm
wasmtime main.wasm
```

## See also

- [Getting Started](getting-started.md)
- [Examples](examples.md)
- [Tensor IR internals](../crates/bhc-tensor-ir/docs/README.md)
- Profile reference on the docs site: https://docs.arcanist.sh/bhc/
