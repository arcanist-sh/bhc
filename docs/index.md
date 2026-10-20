# Basel Haskell Compiler

bhc is a Haskell compiler written in Rust. It compiles Haskell to native
executables through LLVM and to WebAssembly (WASI), and it is built to work with
the [hx](https://arcanist.sh/hx/) toolchain.

These pages are a short companion to the source tree. The full user
documentation lives at **https://docs.arcanist.sh/bhc/**.

## What works today (v0.2.22)

- **Native executables** on Apple Silicon macOS and x86_64 Linux.
- **WebAssembly (WASI)** output with `--target wasm32-wasi`, run with `wasmtime`.
  In the native/WASM differential sweep, 236 of 243 fixtures produce
  byte-identical output on both backends.
- **Real Haskell**: algebraic data types, records, type classes with dictionary
  passing, stock deriving, `Data.Map`, monad transformers such as `StateT`, and
  common GHC extensions (`LambdaCase`, `RecordWildCards`, `TupleSections`, ...).
- **Pandoc as an integration target**: all 221 Pandoc modules pass `bhc check`
  and compile, and `readMarkdown` plus `writeHtml5String` run end to end in
  library mode.
- **Separate compilation** (`-c`, `.bhi` interfaces, `--package-db`) used by
  `hx build --backend bhc`.

## Quick start

```bash
curl -fsSL https://arcanist.sh/bhc/install.sh | sh
export PATH="$HOME/.bhc/bin:$PATH"
export LIBRARY_PATH="$HOME/.bhc/lib:$LIBRARY_PATH"

echo 'main = putStrLn "Hello, World!"' > hello.hs
bhc hello.hs -o hello && ./hello
# Hello, World!
```

See [Getting Started](getting-started.md) for the details, including hx and
building from source.

## Profiles and targets at a glance

```bash
bhc Main.hs -o main                          # native, default profile
bhc --profile numeric Main.hs -o main        # numeric profile (list fusion)
bhc --target wasm32-wasi Main.hs -o main.wasm && wasmtime main.wasm
```

The selectable profiles are `default`, `server`, `numeric`, and `edge`. Compiled
code uses lazy evaluation under every profile. [Profiles](profiles.md) explains
what each one changes today.

## In this folder

- [Getting Started](getting-started.md) - install, first program, hx, REPL
- [Language Guide](language.md) - the Haskell surface bhc accepts, with notes
- [Examples](examples.md) - complete programs, each verified against GHC
- [Profiles](profiles.md) - what `--profile` does today
- [Distribution](DISTRIBUTION.md) - where bhc is published

## Links

- Product site: https://arcanist.sh/bhc/
- Status: https://arcanist.sh/bhc/status/
- Playground: https://arcanist.sh/bhc/playground/
- Documentation: https://docs.arcanist.sh/bhc/
- API reference: https://docs.arcanist.sh/bhc/api/
- Source and issues: https://github.com/arcanist-sh/bhc

## License

bhc is released under the BSD-3-Clause license.
