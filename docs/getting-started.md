# Getting Started with bhc

This page gets you from nothing to a running program. The full guides are the
[quick start](https://docs.arcanist.sh/bhc/docs/quickstart/) and
[installation](https://docs.arcanist.sh/bhc/docs/installation/) pages on
docs.arcanist.sh.

## Install

### Release archive (macOS Apple Silicon, Linux x86_64)

```bash
curl -fsSL https://arcanist.sh/bhc/install.sh | sh
```

The script installs the `bhc` binary and its runtime libraries under `~/.bhc`.
Compiled programs link against those libraries, so put both directories on
your environment:

```bash
export PATH="$HOME/.bhc/bin:$PATH"
export LIBRARY_PATH="$HOME/.bhc/lib:$LIBRARY_PATH"
```

Release archives are published for `aarch64-darwin` and `x86_64-linux` only.
They are also attached to each
[GitHub release](https://github.com/arcanist-sh/bhc/releases).

### With hx

hx can install and drive bhc for a project:

```bash
hx toolchain install --bhc 0.2.22
hx build --backend bhc
```

or select it in `hx.toml`:

```toml
[compiler]
backend = "bhc"

[compiler.bhc]
profile = "default"          # default | server | numeric | edge
# target = "wasm32-wasi"
# emit_kernel_report = true
# tensor_fusion = true
```

### From source

You need the Rust toolchain pinned in `rust-toolchain.toml` and LLVM 21.

```bash
git clone https://github.com/arcanist-sh/bhc.git
cd bhc
cargo build --release -p bhc -p bhc-rts \
  -p bhc-prelude -p bhc-base -p bhc-text -p bhc-containers \
  -p bhc-transformers -p bhc-numeric -p bhc-concurrent -p bhc-system -p bhc-utils
export LIBRARY_PATH="$PWD/target/release:$LIBRARY_PATH"
```

Building only `--bin bhc` leaves out the `libbhc_*.a` runtime and standard
library archives, and then compiled programs fail to link.

### Check the install

```bash
bhc --version
# bhc 0.2.22
```

## Your first program

`hello.hs`:

```haskell
module Main where

main :: IO ()
main = putStrLn "Hello, World!"
```

```bash
bhc hello.hs -o hello
./hello
# Hello, World!
```

Other subcommands:

```bash
bhc check hello.hs     # parse, rename and type-check only
bhc build hello.hs     # same as `bhc hello.hs`; output named after the file
bhc --help             # all flags
```

Put `-o` before the subcommand when you use one (`bhc -o hello build hello.hs`);
the top-level form `bhc hello.hs -o hello` accepts it anywhere.

`bhc run hello.hs [ARGS...]` compiles the program to a temporary executable,
runs it with `ARGS`, and exits with its exit code:

```bash
bhc run hello.hs
# Hello, World!
```

## WebAssembly

```bash
bhc --target wasm32-wasi hello.hs -o hello.wasm
wasmtime hello.wasm
# Hello, World!
```

Spell the target in full: a short triple such as `wasi` currently builds a
native binary ([#42](https://github.com/arcanist-sh/bhc/issues/42)). File IO
goes through the first `--dir` preopen (`wasmtime --dir=. app.wasm`). You can
also try programs in the browser at the
[playground](https://arcanist.sh/bhc/playground/).

## Profiles

```bash
bhc --profile numeric Main.hs -o main
```

The profile is chosen per build, on the command line or in `hx.toml`. There is
no per-module profile pragma. See [Profiles](profiles.md).

## REPL

`bhc repl` starts `bhci`, which must sit next to the `bhc` binary. The release
archives do not include it yet, so the REPL is available from a source build
(`cargo build --release -p bhci`):

```
bhci:001> 2 + 2
4
  :: Int
bhci:002> map (*2) [1, 2, 3]
[2,4,6]
  :: [Int]
bhci:003> :quit
```

## Editor support

The repository contains a language server, `bhc-lsp` (in `crates/bhc-lsp`),
that speaks LSP over stdio. It is not shipped in the release archives; build it
with `cargo build --release -p bhc-lsp` and point your editor's generic LSP
client at the binary.

## Next steps

- [Examples](examples.md) - complete programs with their verified output
- [Language Guide](language.md) - what bhc accepts today
- [Profiles](profiles.md) - what `--profile` changes

## Getting help

- `bhc --help`
- Documentation: https://docs.arcanist.sh/bhc/
- Status page: https://arcanist.sh/bhc/status/
- Issues: https://github.com/arcanist-sh/bhc/issues
