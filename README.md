<div align="center">

<h1 align="center">BHC</h1>

<p align="center">
  <strong>Haskell, made predictable.</strong><br>
  <em>A clean-slate Haskell compiler and runtime for 2026.</em><br>
  <em>Predictable, concurrent, tensor-native.</em>
</p>

<p align="center">
  <a href="https://www.rust-lang.org/">
    <img alt="Rust" src="https://img.shields.io/badge/Rust-stable-000000?logo=rust&logoColor=white&style=for-the-badge">
  </a>
  <a href="https://www.haskell.org/">
    <img alt="Haskell" src="https://img.shields.io/badge/For-Haskell%202026-5e5086?logo=haskell&logoColor=white&style=for-the-badge">
  </a>
  <a href="#license">
    <img alt="License" src="https://img.shields.io/badge/License-BSD--3--Clause-c6a0f6?style=for-the-badge">
  </a>
</p>

<p align="center">
  <a href="https://bhc.raskell.io">Documentation</a> •
  <a href="https://bhc.raskell.io/playground/">Playground</a> •
  <a href="ROADMAP.md">Roadmap</a> •
  <a href="#contributing">Contributing</a>
</p>

</div>

---

GHC is a marvel — and its performance is often folklore. Whether your code fuses, allocates a thunk, or blocks a thread is something you learn by reading Core dumps and staring at profiles, not something the compiler tells you.

**BHC** (the Basel Haskell Compiler) is a clean-slate Haskell compiler and runtime that makes the machine legible. Written in Rust, it keeps the spirit of Haskell while adding explicit performance profiles, a standardized runtime contract, guaranteed fusion, structured concurrency, and a tensor-native numeric pipeline — so behavior is something you can *rely on*, not reverse-engineer.

> **If performance matters, the compiler tells you what happened. If concurrency matters, cancellation is structured. If numerics matter, fusion is guaranteed.**

Named after Basel, Switzerland — a deliberate successor culture to the Glasgow lineage, focused on predictability, concurrency, and numerical computing.

## Why BHC?

- **Predictable by contract** — Explicit profiles give each module a behavioral contract (laziness, strictness, latency, footprint). Performance is correct by construction, not by luck.
- **Guaranteed fusion** — Standard patterns (`map`, `zipWith`, `fold`) fuse into a single loop with no intermediate allocation. In the Numeric Profile, fusion failure is a *compiler bug*, not a surprise.
- **Tensor-native numerics** — A Tensor IR with shape/stride tracking, SIMD auto-vectorization, and BLAS-backed kernels — strict-by-default, unboxed, no hidden thunks.
- **Structured concurrency** — Scoped tasks with automatic cleanup, cooperative cancellation with propagation, deadlines, and a work-stealing scheduler with built-in tracing.
- **Traceable by design** — Kernel reports and fusion diagnostics show exactly what the compiler did. `bhi` lets you inspect every IR stage.
- **A modern runtime contract** — Three explicit memory regions (Hot Arena, Pinned Heap, General Heap) and a generational, incremental GC tuned per profile.
- **Still Haskell** — Multiple editions (Haskell 2010, GHC2021, GHC2024, H26) and 30+ extensions, so existing code feels at home.
- **Part of the toolchain** — BHC is the next-generation backend behind [**hx**](https://github.com/arcanist-sh/hx), the unified Haskell toolchain: `hx` drives BHC the same way it drives GHC.

## Profiles

BHC compiles every module under an explicit profile with a distinct performance contract:

| Profile | Use Case | Key Characteristics |
|---------|----------|---------------------|
| **Default** | General Haskell | Lazy evaluation, GC managed |
| **Server** | Web services, daemons | Structured concurrency, bounded latency, observability |
| **Numeric** | ML, linear algebra, tensors | Strict-by-default, unboxed, fusion guaranteed |
| **Edge** | Embedded, WASM | Minimal runtime footprint |

Profiles are explicit and localizable — per package or per module:

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
module HotPath where
```

## Conformance Levels

BHC targets the Haskell 2026 Platform specification:

- **H26-Core** — Language core + minimal runtime contract
- **H26-Platform** — Core + standard libraries + packaging
- **H26-Numeric** — Platform + Numeric Profile + Tensor IR guarantees

## CLI Tools

| Command | Description |
|---------|-------------|
| `bhc` | Compiler driver |
| `bhci` | Interactive REPL |
| `bhi` | IR inspector / kernel reports |

## Quick Start

```bash
# Build the compiler
cargo build --release

# Compile a program
./target/release/bhc hello.hs -o hello

# Run with the Numeric Profile
./target/release/bhc --profile=numeric matmul.hs -o matmul

# See exactly what the compiler did
./target/release/bhc --profile=numeric --kernel-report tensor_ops.hs
```

> **Try it in your browser:** [bhc.raskell.io/playground](https://bhc.raskell.io/playground/)

## Example

```haskell
{-# HASKELL_EDITION 2026 #-}
{-# PROFILE Numeric #-}

module Main where

import H26.Tensor

-- Dot product: guaranteed to fuse into a single loop
dot :: Tensor Float -> Tensor Float -> Float
dot xs ys = sum (zipWith (*) xs ys)

-- Matrix multiply: auto-vectorized, parallel
matmul :: Tensor Float -> Tensor Float -> Tensor Float
matmul a b = parMap (\i ->
    parMap (\j -> dot (row i a) (col j b)) [0..n-1]
  ) [0..m-1]
  where
    (m, _) = shape a
    (_, n) = shape b

main :: IO ()
main = do
  let a = fromList [2, 3] [1, 2, 3, 4, 5, 6]
      b = fromList [3, 2] [1, 2, 3, 4, 5, 6]
      c = matmul a b
  print c
```

## Project Structure

```
bhc/
├── crates/                    # Rust compiler implementation
│   ├── bhc/                   # Main CLI binary
│   ├── bhc-driver/            # Compilation orchestration
│   ├── bhc-parser/            # Parsing (lexer, AST)
│   ├── bhc-typeck/            # Type inference & checking
│   ├── bhc-core/              # Core IR + interpreter
│   ├── bhc-tensor-ir/         # Tensor IR (Numeric profile)
│   ├── bhc-loop-ir/           # Loop IR (vectorization)
│   ├── bhc-codegen/           # Native code generation (LLVM)
│   ├── bhc-wasm/              # WebAssembly backend
│   ├── bhc-gpu/               # GPU backends (CUDA/ROCm)
│   └── bhc-playground/        # Browser WASM playground
├── rts/                       # Runtime system (Rust)
│   ├── bhc-rts/               # Core runtime
│   └── bhc-rts-gc/            # Garbage collector
├── stdlib/                    # Standard library
│   ├── bhc-prelude/           # Prelude primitives
│   ├── bhc-base/              # Base library
│   ├── bhc-containers/        # Data structures
│   ├── bhc-numeric/           # Numeric/SIMD/BLAS
│   └── H26/                   # H26 Platform modules
├── tools/                     # Additional tools
│   ├── bhci/                  # Interactive REPL
│   ├── bhi/                   # IR inspector
│   └── bhc-docs/              # Documentation generator
└── tests/                     # Test suites
```

## Roadmap

| Milestone | Name | Status |
|-----------|------|--------|
| Phase 1 | Native Hello World | ✅ Complete |
| Phase 2 | Language Completeness | ✅ Complete |
| Phase 3 | Numeric Profile | ✅ Complete |
| Phase 4 | WASM Backend | 🟡 70% |
| Phase 5 | Server Profile | ✅ Complete |
| Phase 6 | GPU Backend | 🟡 80% |

See [ROADMAP.md](ROADMAP.md) for detailed milestone specifications.

## Documentation

- [Website](https://bhc.raskell.io) — Official site with guides and tutorials
- [API Docs](https://bhc.raskell.io/docs/api/) — Standard library reference (63 modules)
- [Playground](https://bhc.raskell.io/playground/) — Try BHC in your browser
- [ROADMAP.md](ROADMAP.md) — Implementation status and milestones
- [.claude/CLAUDE.md](.claude/CLAUDE.md) — Development guidelines

## Building

### Prerequisites

- Rust 1.82+ (stable toolchain)
- LLVM 21 (for native codegen)
- `wasm32-unknown-unknown` target (for the playground)

### Build Commands

```bash
# Build everything
cargo build

# Build release
cargo build --release

# Run tests
cargo test

# Run a specific crate's tests
cargo test -p bhc-parser

# Run benchmarks
cargo bench

# Build and run bhc
cargo run --bin bhc -- Main.hs
```

## Philosophy

BHC prioritizes **predictability over folklore**. Three principles follow from that:

1. **Legible over magic** — The compiler reports what it did (kernel reports, fusion diagnostics, allocation tracking) instead of leaving you to infer it from profiles.
2. **Contracts over guesswork** — Profiles make performance behavior explicit and local, so a module's costs are part of its interface.
3. **Correct by construction** — Guaranteed fusion, structured cancellation, and a standardized runtime contract make the fast, safe path the default — not a reward for expertise.

The destination: a Haskell that is **predictable, concurrent, and tensor-native** — without giving up purity, and without asking you to read Core to trust your code.

## Contributing

Contributions are welcome. Please read the guidelines in `.claude/rules/` before submitting changes.

### Commit Messages

Use conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `perf:` Performance improvements
- `refactor:` Code restructuring
- `docs:` Documentation
- `test:` Test additions/changes

## License

BSD-3-Clause — part of [arcanist.sh](https://arcanist.sh).

## Acknowledgments

BHC builds on decades of research in functional programming, type systems, and compiler construction. We acknowledge the foundational work of the GHC team and the broader Haskell community.
