<div align="center">

<h1 align="center">BHC</h1>

<p align="center">
  <strong>An alternative compiler for Haskell.</strong><br>
  <em>Compatibility-first, with runtime profiles and a tensor-native numeric pipeline.</em><br>
  <em>Same language, more options.</em>
</p>

<p align="center">
  <a href="https://www.rust-lang.org/">
    <img alt="Rust" src="https://img.shields.io/badge/Rust-stable-000000?logo=rust&logoColor=white&style=for-the-badge">
  </a>
  <a href="https://www.haskell.org/">
    <img alt="Haskell" src="https://img.shields.io/badge/For-Haskell-5e5086?logo=haskell&logoColor=white&style=for-the-badge">
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

**BHC** (the Basel Haskell Compiler) is an alternative, clean-slate compiler and runtime for Haskell, written in Rust. It treats compatibility with the language as the baseline, then adds new capabilities on top — strictly opt-in.

Those capabilities are explicit runtime profiles, a standardized runtime contract, guaranteed fusion patterns, structured concurrency, a tensor-native numeric pipeline, and compilation targets beyond native binaries (WebAssembly, GPU). None of them change standard semantics unless you ask for them.

> **Compatibility as the baseline. Performance and new targets are opt-in — explicit, namespaced, and documented per release.**

BHC is named after Basel, Switzerland. It is an independent implementation: it does not define the Haskell standard and does not aim to replace GHC — it aims to give Haskell more places to run and more performance contracts to choose from.

## Why BHC?

- **Compatibility-first** — Standard Haskell is the baseline. BHC compiles existing code and aims to match its semantics; new behavior is opt-in and namespaced (`BHC.*` extensions, the `BHC2026` bundle), so nothing changes unless you enable it.
- **Runtime profiles** — `default`, `server`, `numeric`, and `edge`, each with an explicit behavioral contract (laziness, latency, footprint). Selectable per package or per module.
- **Transparent performance** — Guaranteed fusion patterns for standard containers and reduced heap activity in hot loops. Kernel reports and the `bhi` inspector show exactly what the compiler did — no guessing from profiles.
- **Tensor-native numerics** — A Tensor IR with shape/stride tracking, SIMD vectorization, and BLAS-backed kernels; strict-by-default and unboxed in the Numeric profile.
- **Structured concurrency** — Scoped tasks with automatic cleanup, cooperative cancellation with propagation, deadlines, and a work-stealing scheduler with tracing (Server profile).
- **A modern runtime contract** — Three explicit memory regions (Hot Arena, Pinned Heap, General Heap) and a generational, incremental GC tuned per profile.
- **Portable where it matters** — Native code today via LLVM; WebAssembly and GPU backends in progress.
- **Part of the toolchain** — BHC is a first-class backend for [**hx**](https://github.com/arcanist-sh/hx), the Haskell toolchain: `hx` can drive BHC the same way it drives GHC.

## Profiles

BHC compiles every module under an explicit profile with its own performance contract:

| Profile | Use Case | Key Characteristics |
|---------|----------|---------------------|
| **default** | General Haskell | Correctness and compatibility first; lazy, GC-managed |
| **server** | Web services, daemons | Structured concurrency, predictable latency, observability |
| **numeric** | ML, linear algebra, tensors | Strict-by-default, unboxed, guaranteed fusion |
| **edge** | Embedded, WASM | Minimal runtime footprint |

Profiles are explicit and local — per package or per module:

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
module HotPath where
```

## Compatibility

Compatibility is the baseline, not an afterthought. BHC tracks, per release, which language features compile and match GHC semantics, which compile with minor changes, which differ at the runtime level, and which aren't supported yet.

BHC-specific behavior is always opt-in and namespaced — `{-# LANGUAGE BHC.TensorIR #-}`, the `BHC2026` bundle pragma — so it never silently diverges from standard Haskell. See the [compatibility charter](https://bhc.raskell.io) for the current matrix.

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

# Compile under the numeric profile
./target/release/bhc --profile=numeric matmul.hs -o matmul

# See what the compiler did
./target/release/bhc --profile=numeric --kernel-report tensor_ops.hs
```

> **Try it in your browser:** [bhc.raskell.io/playground](https://bhc.raskell.io/playground/)

## Example

```haskell
{-# OPTIONS_BHC -profile=numeric #-}

module Main where

import H26.Tensor

-- Dot product: a guaranteed fusion pattern — compiles to a single loop
dot :: Tensor Float -> Tensor Float -> Float
dot xs ys = sum (zipWith (*) xs ys)

-- Matrix multiply: vectorized and parallel
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
│   ├── bhc-tensor-ir/         # Tensor IR (numeric profile)
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
│   └── bhc-numeric/           # Numeric/SIMD/BLAS
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

See [ROADMAP.md](ROADMAP.md) for detailed milestone specifications. BHC is under active development — some language features, libraries, and targets are still incomplete.

## Documentation

- [Website](https://bhc.raskell.io) — Official site with guides and tutorials
- [API Docs](https://bhc.raskell.io/docs/api/) — Standard library reference
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

BHC is a serious systems project with explicit boundaries:

1. **Compatibility is the contract** — Standard Haskell is the baseline BHC measures itself against, tracked release to release.
2. **Innovation is opt-in** — New runtime behavior, profiles, and extensions are enabled explicitly and namespaced, never imposed.
3. **Performance is transparent** — When it matters, the compiler reports what happened (kernel reports, fusion diagnostics, allocation tracking) instead of leaving you to infer it.

The destination: a Haskell that is **portable, predictable, and tensor-native** — without splitting the ecosystem or asking you to leave the language behind.

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

BHC is an independent compiler implementation for Haskell. "Haskell" is a community language; BHC does not define the Haskell standard. BHC builds on decades of research in functional programming, type systems, and compiler construction, and we acknowledge the foundational work of the GHC team and the broader Haskell community.
