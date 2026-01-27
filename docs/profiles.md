# Runtime Profiles

BHC's profile system lets you choose the runtime behavior that best fits your application.

## Overview

| Profile | Evaluation | GC | Fusion | SIMD | Use Case |
|---------|------------|-----|--------|------|----------|
| `default` | Lazy | Standard | Opportunistic | No | General Haskell |
| `server` | Lazy | Incremental | Opportunistic | No | Web services |
| `numeric` | Strict | Minimal | Guaranteed | Yes | ML, linear algebra |
| `edge` | Lazy | Minimal | Opportunistic | No | WASM, serverless |
| `realtime` | Lazy | Bounded | Opportunistic | No | Games, audio |
| `embedded` | Strict | None | Opportunistic | Optional | Microcontrollers |

## Default Profile

The standard Haskell experience with lazy evaluation.

### Characteristics

- **Lazy evaluation** - Expressions evaluated only when needed
- **Standard GC** - Generational garbage collector
- **Opportunistic fusion** - Fusion applied where detected

### When to Use

- General-purpose Haskell code
- When you want standard Haskell semantics
- When lazy evaluation is beneficial

### Example

```haskell
-- Lazy: only computes as much as needed
takeWhile (< 10) [1..]  -- Works on infinite list!
```

## Server Profile

Optimized for web services and long-running applications.

### Characteristics

- **Structured concurrency** - `withScope`, `spawn`, `await` primitives
- **Cooperative cancellation** - Tasks check for cancellation at safe points
- **Incremental GC** - Lower pause times
- **Deadlines** - Built-in timeout support

### Key APIs

```haskell
-- Structured concurrency
withScope :: (Scope -> IO a) -> IO a
spawn :: Scope -> IO a -> IO (Task a)
await :: Task a -> IO a
cancel :: Task a -> IO ()

-- Deadlines
withDeadline :: Duration -> (Scope -> IO a) -> IO (Maybe a)
```

### Example

```haskell
{-# OPTIONS_BHC -profile=server #-}
module Server where

handleRequest :: Request -> IO Response
handleRequest req = withDeadline (seconds 30) $ \scope -> do
    -- These run in parallel
    user <- spawn scope $ fetchUser (reqUserId req)
    posts <- spawn scope $ fetchPosts (reqUserId req)

    -- Wait for both
    u <- await user
    p <- await posts

    pure (Response u p)
```

## Numeric Profile

High-performance numeric computing with guaranteed optimization.

### Characteristics

- **Strict evaluation** - No lazy thunks in numeric code
- **Guaranteed fusion** - Specified patterns MUST fuse
- **SIMD vectorization** - Automatic use of AVX/SSE
- **Arena allocation** - Kernel temporaries in hot arena
- **Unboxed by default** - No boxing overhead for primitives

### Guaranteed Fusion Patterns

These patterns fuse into single passes with no intermediate allocation:

```haskell
-- Pattern 1: Map composition
map f (map g xs)           -- -> map (f . g) xs

-- Pattern 2: ZipWith with maps
zipWith f (map g a) (map h b)  -- -> single traversal

-- Pattern 3: Fold of map
sum (map f xs)             -- -> single traversal

-- Pattern 4: Strict fold of map
foldl' op z (map f xs)     -- -> single traversal
```

### Fusion Verification

```bash
# Generate fusion report
bhc --profile=numeric --kernel-report matrix.hs

# Output:
# [Kernel k1] dotProduct: FUSED
#   Pattern: sum/zipWith
#   Single traversal, SIMD width: 8 x f32
#   Allocation: None
```

### Example

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
module Matrix where

-- Guaranteed to fuse into single SIMD loop
dotProduct :: [Double] -> [Double] -> Double
dotProduct xs ys = sum (zipWith (*) xs ys)

-- Matrix operations fuse
matmul :: Matrix Double -> Matrix Double -> Matrix Double
matmul a b = ...  -- Tiled, cache-aware, SIMD-optimized
```

## Edge Profile

Minimal runtime for WebAssembly and serverless.

### Characteristics

- **Minimal footprint** - Small binary size
- **No runtime threads** - Single-threaded execution
- **Simplified GC** - Basic collector or reference counting
- **Fast startup** - Quick initialization

### When to Use

- WebAssembly targets (`--target=wasi`)
- AWS Lambda / serverless functions
- Browser applications

### Example

```haskell
{-# OPTIONS_BHC -profile=edge #-}
module Lambda where

-- Minimal dependencies, fast startup
handler :: Event -> IO Response
handler event = do
    let result = processEvent event
    pure (Response 200 result)
```

## Realtime Profile

Bounded GC pauses for interactive applications.

### Characteristics

- **Bounded GC pauses** - Maximum pause configurable (default: 1ms)
- **Per-frame arenas** - Temporary allocations freed per frame
- **Incremental collection** - Work spread across frames

### Configuration

```toml
[profile.realtime]
max_gc_pause_ms = 1
frame_arena_size = "4MB"
```

### Example

```haskell
{-# OPTIONS_BHC -profile=realtime #-}
module Game where

gameLoop :: GameState -> IO ()
gameLoop state = do
    withFrameArena $ do  -- Arena freed after frame
        input <- pollInput
        let state' = updateGame state input
        renderGame state'
        gameLoop state'
```

## Embedded Profile

No garbage collector, static allocation only.

### Characteristics

- **No GC** - All memory statically allocated
- **Strict evaluation** - No thunks
- **Compile-time bounds** - Array sizes known at compile time
- **No runtime allocation** - Stack + static only

### Restrictions

- No recursive data types
- No arbitrary-precision Integer
- Fixed-size arrays only
- No dynamic allocation

### Example

```haskell
{-# OPTIONS_BHC -profile=embedded #-}
module Sensor where

-- Fixed-size buffer, no allocation
data SensorBuffer = SensorBuffer (Array 256 Float)

readSensor :: IO Float
readSensor = primReadSensor 0x4000

processReading :: Float -> SensorBuffer -> SensorBuffer
processReading val buf = shiftIn val buf
```

## Profile Selection

### Command Line

```bash
bhc --profile=numeric myfile.hs
```

### Per-Module Pragma

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
module HotPath where
```

### In bhc.toml

```toml
[build]
profile = "default"

[modules."HotPath"]
profile = "numeric"
```

## Profile Compatibility

Modules with different profiles can be mixed, with some restrictions:

| Caller Profile | Callee Profile | Allowed? |
|----------------|----------------|----------|
| default | numeric | Yes |
| numeric | default | Yes (with boxing) |
| server | numeric | Yes |
| embedded | default | No |
| realtime | numeric | Yes |

## See Also

- [Language Guide](language.md) - Haskell syntax reference
- [Examples](examples.md) - Profile-specific examples
- [Tensor IR](../crates/bhc-tensor-ir/docs/README.md) - Numeric internals
