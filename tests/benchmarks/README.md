# BHC Benchmarks

Performance benchmarks for the BHC compiler and runtime.

## Numeric Benchmarks (H26-Numeric Required)

These benchmarks are required for H26-Numeric conformance:

| Benchmark | Description | Requirement |
|-----------|-------------|-------------|
| dot | Dot product | Fused, vectorized |
| saxpy | a*x + y | Fused, vectorized |
| matmul | Matrix multiplication | Auto-vectorized |
| sum | Array reduction | Fused with maps |
| max | Array reduction | Fused with maps |

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- dot
cargo bench -- matmul

# With detailed output
cargo bench -- --verbose
```

## Benchmark Results

Results are published to `target/criterion/` with HTML reports.

## Performance Requirements

Per H26-SPEC Section 13.3:

1. Fused kernels MUST NOT allocate on General Heap
2. Vectorization MUST occur on supported targets
3. Parallel reductions MUST scale linearly
