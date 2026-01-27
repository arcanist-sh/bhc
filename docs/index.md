# Basel Haskell Compiler

**BHC** is a modern Haskell compiler and runtime designed for the 2026 era of functional programming.

## Why BHC?

- **Predictable Performance** - No hidden allocations, guaranteed fusion, transparent costs
- **Modern Profiles** - Choose between lazy, strict, numeric, or realtime semantics
- **Multi-Target** - Compile to native code, WebAssembly, or GPU
- **Excellent Tooling** - LSP support, interactive REPL, IR inspector

## Quick Start

```bash
# Install BHC
curl -fsSL https://bhc.raskell.io/install.sh | sh

# Create your first program
echo 'main = putStrLn "Hello, World!"' > hello.hs

# Compile and run
bhc hello.hs -o hello && ./hello
```

## Features

### Profiles

BHC supports multiple runtime profiles to match your use case:

| Profile | Use Case | Evaluation |
|---------|----------|------------|
| `default` | General Haskell | Lazy |
| `server` | Web services | Lazy + Structured concurrency |
| `numeric` | ML, linear algebra | Strict + Guaranteed fusion |
| `edge` | WASM, serverless | Minimal footprint |
| `realtime` | Games, audio | Bounded GC pauses |
| `embedded` | Microcontrollers | No GC |

```bash
# Use numeric profile for fast computations
bhc --profile=numeric matrix.hs
```

### Targets

Compile to multiple backends:

```bash
# Native (default)
bhc main.hs -o main

# WebAssembly
bhc --target=wasi main.hs -o main.wasm

# GPU (CUDA)
bhc --target=cuda --profile=numeric compute.hs
```

### Guaranteed Fusion

In the `numeric` profile, these patterns always fuse into single passes:

```haskell
-- Single loop, no intermediate arrays
result = sum (map (*2) (filter even xs))

-- Fused matrix operations
output = matmul A (matmul B C)
```

## Documentation

- [Getting Started](getting-started.md) - Installation and first steps
- [Language Guide](language.md) - Haskell language reference
- [Profiles](profiles.md) - Runtime profile system
- [Examples](examples.md) - Code examples and tutorials

## Compatibility

BHC supports multiple Haskell editions:

- Haskell 2010
- GHC2021
- GHC2024
- Haskell 2026 (H26) - default

Most existing GHC code will work with minimal changes.

## Community

- [GitHub](https://github.com/raskell-io/bhc) - Source code and issues
- [Discord](https://discord.gg/bhc) - Community chat
- [Forum](https://forum.raskell.io) - Discussions

## License

BHC is released under the BSD-3-Clause license.
