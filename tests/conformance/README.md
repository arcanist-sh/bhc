# BHC Conformance Test Suite

This directory contains the official H26 Platform conformance tests.

## Test Categories

### Semantic Tests (`semantic/`)

Tests for language semantics:
- Strictness behavior per profile
- Exception propagation
- Determinism requirements

### Runtime Tests (`runtime/`)

Tests for runtime system behavior:
- Cancellation propagation
- Structured concurrency correctness
- Pinned allocation immovability
- Atomic memory ordering

### Numeric Benchmarks (`numeric/`)

Performance tests for H26-Numeric conformance:
- dot product
- saxpy
- matmul (small/medium/large)
- reduction (sum/max)
- fusion scenarios

## Running Tests

```bash
# Run all conformance tests
cargo test --test conformance

# Run specific category
cargo test --test conformance -- semantic
cargo test --test conformance -- runtime
cargo test --test conformance -- numeric
```

## Adding Tests

Each test file should follow this structure:

```haskell
-- Test: <test-name>
-- Category: <semantic|runtime|numeric>
-- Profile: <default|server|numeric|edge>
-- Expected: <success|error <code>|output <expected>>
-- Spec: H26-SPEC Section X.Y

module Test where

-- Test code here
```
