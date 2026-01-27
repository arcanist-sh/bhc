# Getting Started with BHC

This guide will help you install BHC and write your first program.

## Installation

### Linux / macOS

```bash
curl -fsSL https://bhc.raskell.io/install.sh | sh
```

### Windows

```powershell
iwr -useb https://bhc.raskell.io/install.ps1 | iex
```

### From Source

```bash
git clone https://github.com/raskell-io/bhc.git
cd bhc
cargo build --release
cargo install --path crates/bhc
```

### Verify Installation

```bash
bhc --version
# bhc 0.2.1
```

## Your First Program

Create a file called `hello.hs`:

```haskell
module Main where

main :: IO ()
main = putStrLn "Hello, World!"
```

Compile and run:

```bash
bhc hello.hs -o hello
./hello
# Hello, World!
```

## Using the REPL

Start the interactive REPL:

```bash
bhci
```

Try some expressions:

```
bhci> 2 + 2
4

bhci> map (*2) [1, 2, 3]
[2, 4, 6]

bhci> :type foldl
foldl :: (b -> a -> b) -> b -> [a] -> b

bhci> :quit
```

## Project Setup

Initialize a new project:

```bash
bhc init my-project
cd my-project
```

This creates:

```
my-project/
├── bhc.toml         # Project configuration
├── src/
│   └── Main.hs      # Entry point
└── tests/
    └── Test.hs      # Test module
```

Build and run:

```bash
bhc build
bhc run
```

## Project Configuration

The `bhc.toml` file configures your project:

```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "H26"

[build]
profile = "default"
optimization = 2

[dependencies]
base = "1.0"
containers = "0.6"
```

## Using Profiles

### Default Profile

Standard lazy Haskell semantics:

```bash
bhc hello.hs
```

### Numeric Profile

For performance-critical numeric code:

```bash
bhc --profile=numeric matrix.hs
```

Guarantees:
- Strict evaluation
- Guaranteed fusion
- SIMD vectorization
- No intermediate allocations

Example:

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
module DotProduct where

dotProduct :: [Double] -> [Double] -> Double
dotProduct xs ys = sum (zipWith (*) xs ys)
-- Fuses into a single SIMD loop!
```

### Server Profile

For web services with structured concurrency:

```bash
bhc --profile=server server.hs
```

Example:

```haskell
{-# OPTIONS_BHC -profile=server #-}
module Server where

import Control.Concurrent.Scope

handleRequest :: Request -> IO Response
handleRequest req = withDeadline (seconds 30) $ \scope -> do
    user <- spawn scope $ fetchUser (reqUserId req)
    data <- spawn scope $ fetchData (reqDataId req)
    await user >>= \u -> await data >>= \d -> pure (Response u d)
```

## Editor Setup

### VS Code

Install the "BHC" extension from the marketplace.

### Neovim

Add to your config:

```lua
require('lspconfig').bhc.setup {
  cmd = { "bhc-lsp" },
  filetypes = { "haskell" },
}
```

### Emacs

```elisp
(add-to-list 'lsp-language-id-configuration '(haskell-mode . "haskell"))
(lsp-register-client
  (make-lsp-client :new-connection (lsp-stdio-connection "bhc-lsp")
                   :major-modes '(haskell-mode)
                   :server-id 'bhc-lsp))
```

## Next Steps

- [Language Guide](language.md) - Learn Haskell with BHC
- [Profiles](profiles.md) - Deep dive into profiles
- [Examples](examples.md) - More code examples

## Getting Help

- `bhc --help` - Command-line help
- `/help` in the REPL - REPL commands
- [GitHub Issues](https://github.com/raskell-io/bhc/issues) - Bug reports
- [Discord](https://discord.gg/bhc) - Community support
