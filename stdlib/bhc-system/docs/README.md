# BHC System Library

System-level I/O operations for BHC.

## Modules

| Module | Description |
|--------|-------------|
| `BHC.System.IO` | File handles, buffered I/O, console operations |
| `BHC.System.Process` | Process spawning and management |
| `BHC.System.Environment` | Environment variables, program arguments |
| `BHC.System.Directory` | Directory operations, file queries |
| `BHC.System.FilePath` | Path manipulation utilities |

## Quick Start

```haskell
import BHC.System.IO
import BHC.System.Environment

main :: IO ()
main = do
    args <- getArgs
    contents <- readFile (head args)
    putStrLn contents
```

## Features

- **Handle-based I/O**: Full control over file handles with buffering modes
- **Process Management**: Spawn and control child processes
- **Path Operations**: Cross-platform filepath manipulation
- **Directory Queries**: Check existence, list contents, get metadata

## See Also

- [DESIGN.md](DESIGN.md) - Design decisions
- [BHC.Prelude](../../bhc-prelude/docs/README.md) - Core prelude
