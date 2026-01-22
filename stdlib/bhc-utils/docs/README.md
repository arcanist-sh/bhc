# BHC Utils Library

Common utility modules for BHC applications.

## Modules

| Module | Description |
|--------|-------------|
| `BHC.Data.Time` | Dates, times, durations, instants |
| `BHC.Data.Random` | Random number generation and distributions |
| `BHC.Data.JSON` | JSON parsing and serialization |

## Quick Start

### Time

```haskell
import BHC.Data.Time

main :: IO ()
main = do
    start <- now
    -- ... computation ...
    elapsed <- elapsed start
    putStrLn $ "Took " ++ show (toMillis elapsed) ++ "ms"
```

### Random

```haskell
import BHC.Data.Random

main :: IO ()
main = do
    rng <- newRng
    n <- range rng 1 100
    print n
```

### JSON

```haskell
import BHC.Data.JSON

data User = User { name :: String, age :: Int }

instance ToJSON User where
    toJSON (User n a) = object ["name" .= n, "age" .= a]

instance FromJSON User where
    fromJSON = withObject $ \o -> 
        User <$> o .: "name" <*> o .: "age"
```

## Features

- **Time**: ISO 8601 parsing, duration arithmetic, benchmarking utilities
- **Random**: Cryptographically seeded, multiple distributions (uniform, normal, exponential)
- **JSON**: Type-safe encoding/decoding with `ToJSON`/`FromJSON` classes

## Performance

| Operation | Size | Time |
|-----------|------|------|
| JSON parse | 1KB | ~50μs |
| JSON encode | 1KB | ~30μs |
| Random int | 1 | ~10ns |

## See Also

- [DESIGN.md](DESIGN.md) - Design decisions
- [BENCHMARKS.md](BENCHMARKS.md) - Performance data
