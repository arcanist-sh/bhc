# bhc-lsp

Language Server Protocol implementation for the Basel Haskell Compiler.

## Overview

The `bhc-lsp` crate provides IDE support through the Language Server Protocol (LSP), enabling features like:

- Real-time diagnostics (errors, warnings)
- Go to definition
- Find references
- Hover information
- Code completion
- Document symbols
- Workspace symbols
- Code formatting

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Editor                               │
│              (VS Code, Neovim, Emacs, etc.)                 │
└────────────────────────┬────────────────────────────────────┘
                         │ JSON-RPC (stdio)
┌────────────────────────▼────────────────────────────────────┐
│                      bhc-lsp                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Server    │  │  Document   │  │     Analysis        │  │
│  │  Main Loop  │  │   Manager   │  │      Engine         │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         ▼                ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Request Handlers                    │    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐   │    │
│  │  │ Hover  │ │ DefRef │ │ Compl. │ │ Formatting │   │    │
│  │  └────────┘ └────────┘ └────────┘ └────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `server` | Main LSP server loop and message routing |
| `capabilities` | Server capability declarations |
| `document` | Document management with rope-based editing |
| `analysis` | Text-based symbol extraction and analysis |
| `handlers` | Request handlers for LSP methods |
| `completion` | Code completion with keywords, builtins, snippets |
| `hover` | Hover information for symbols |
| `navigation` | Go-to-definition and find-references |
| `symbols` | Document and workspace symbol providers |
| `diagnostics` | Diagnostic creation helpers |
| `config` | Server configuration management |

## Supported LSP Methods

### Document Synchronization

| Method | Support |
|--------|---------|
| `textDocument/didOpen` | ✅ |
| `textDocument/didChange` | ✅ |
| `textDocument/didClose` | ✅ |
| `textDocument/didSave` | ✅ |

### Language Features

| Method | Support |
|--------|---------|
| `textDocument/hover` | ✅ |
| `textDocument/completion` | ✅ |
| `textDocument/definition` | ✅ |
| `textDocument/references` | ✅ |
| `textDocument/documentSymbol` | ✅ |
| `textDocument/formatting` | ✅ (placeholder) |
| `textDocument/codeAction` | 🔜 |
| `textDocument/rename` | 🔜 |

### Workspace Features

| Method | Support |
|--------|---------|
| `workspace/symbol` | ✅ |
| `workspace/didChangeConfiguration` | ✅ |

## Usage

### Running the Server

```bash
# Start the LSP server (communicates via stdio)
bhc-lsp
```

### Editor Configuration

#### VS Code

```json
{
  "bhc.server.path": "/path/to/bhc-lsp",
  "bhc.trace.server": "verbose"
}
```

#### Neovim (with nvim-lspconfig)

```lua
require('lspconfig').bhc.setup {
  cmd = { "bhc-lsp" },
  filetypes = { "haskell" },
  root_dir = require('lspconfig.util').root_pattern("bhc.toml", "*.cabal", "stack.yaml"),
}
```

#### Emacs (with lsp-mode)

```elisp
(lsp-register-client
 (make-lsp-client
  :new-connection (lsp-stdio-connection "bhc-lsp")
  :major-modes '(haskell-mode)
  :server-id 'bhc-lsp))
```

## Configuration

The server accepts configuration via the `initialize` request:

```json
{
  "initializationOptions": {
    "diagnostics": {
      "enabled": true,
      "onSave": true,
      "onType": true
    },
    "completion": {
      "enabled": true,
      "maxResults": 100,
      "snippets": true
    },
    "formatting": {
      "enabled": true,
      "indentWidth": 2,
      "useSpaces": true
    }
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `diagnostics.enabled` | bool | `true` | Enable diagnostics |
| `diagnostics.onSave` | bool | `true` | Run diagnostics on save |
| `diagnostics.onType` | bool | `true` | Run diagnostics while typing |
| `completion.enabled` | bool | `true` | Enable completion |
| `completion.maxResults` | int | `100` | Maximum completion results |
| `completion.snippets` | bool | `true` | Include snippet completions |
| `formatting.enabled` | bool | `true` | Enable formatting |
| `formatting.indentWidth` | int | `2` | Indentation width |
| `formatting.useSpaces` | bool | `true` | Use spaces for indentation |

## Completion Features

### Keywords

Common Haskell keywords with descriptions:
- `module`, `import`, `where`, `let`, `in`
- `if`, `then`, `else`, `case`, `of`
- `data`, `newtype`, `type`, `class`, `instance`
- `do`, `forall`, `deriving`

### Builtin Functions

~50 commonly used Prelude functions:
- List: `map`, `filter`, `foldl`, `foldr`, `head`, `tail`, `length`
- IO: `putStrLn`, `print`, `getLine`, `readFile`
- Functional: `id`, `const`, `flip`, `compose`

### Snippets

Code snippets for common patterns:
- `main` - Main function template
- `module` - Module declaration
- `data` - Data type declaration
- `class` - Type class declaration
- `case` - Case expression
- `do` - Do notation block

## Implementation Details

### Document Management

Documents are stored using the `ropey` crate for efficient text manipulation:

```rust
pub struct Document {
    pub uri: Uri,
    pub content: Rope,
    pub version: i32,
}
```

The rope data structure provides:
- O(log n) character indexing
- O(log n) line indexing
- Efficient incremental updates

### Analysis Engine

The analysis engine extracts symbols using text pattern matching:

```rust
pub struct AnalysisResult {
    pub diagnostics: Vec<Diagnostic>,
    pub symbols: Vec<Symbol>,
}

pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub range: Range,
    pub type_sig: Option<String>,
}
```

Symbol extraction recognizes:
- Module declarations: `module Foo where`
- Function definitions: `foo x y = ...`
- Type signatures: `foo :: Int -> Int`
- Data types: `data Foo = ...`
- Type aliases: `type Foo = ...`
- Type classes: `class Foo a where`

### Concurrent Access

The server uses `dashmap` for thread-safe caching:

```rust
pub struct AnalysisEngine {
    cache: DashMap<String, Arc<AnalysisResult>>,
}
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `lsp-server` | LSP protocol transport |
| `lsp-types` | LSP type definitions |
| `ropey` | Efficient text rope |
| `dashmap` | Concurrent hash map |
| `crossbeam-channel` | Message passing |
| `tracing` | Logging |
| `serde_json` | JSON serialization |

## Testing

```bash
# Run unit tests
cargo test -p bhc-lsp

# Run with verbose logging
RUST_LOG=debug bhc-lsp
```

## Future Work

- Full integration with `bhc-parser` and `bhc-typeck` for semantic analysis
- Code actions (quick fixes, refactoring)
- Rename symbol support
- Call hierarchy
- Semantic highlighting
- Inlay hints
- Code lens

## See Also

- [Language Server Protocol Specification](https://microsoft.github.io/language-server-protocol/)
- [bhc-driver](../../bhc-driver/docs/README.md) - Compilation driver
- [bhc-parser](../../bhc-parser/docs/README.md) - Parser implementation
- [bhc-typeck](../../bhc-typeck/docs/README.md) - Type checker
