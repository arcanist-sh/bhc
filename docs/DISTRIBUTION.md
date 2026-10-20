# bhc distribution channels

Where bhc is published, and what each channel installs. Current release:
v0.2.22.

## Recommended

| Channel | Install | Notes |
|---------|---------|-------|
| Install script | `curl -fsSL https://arcanist.sh/bhc/install.sh \| sh` | Installs to `~/.bhc`; set `PATH=~/.bhc/bin` and `LIBRARY_PATH=~/.bhc/lib` |
| hx | `hx toolchain install --bhc 0.2.22` | Then `hx build --backend bhc`, or `[compiler] backend = "bhc"` in `hx.toml` |
| GitHub Releases | [releases](https://github.com/arcanist-sh/bhc/releases) | `.tar.gz` per platform with `bhc` and `lib/libbhc_*.a`, plus `SHA256SUMS.txt` |
| From source | `cargo build --release -p bhc -p bhc-rts ...` | See [Getting Started](getting-started.md#from-source); needs LLVM 21 |

The install script does not yet check downloads against `SHA256SUMS.txt`
([#46](https://github.com/arcanist-sh/bhc/issues/46)); verify by hand if you
need to.

## Other channels

| Channel | State |
|---------|-------|
| Homebrew (`arcanist-sh/tap/bhc`) | Formula is stuck at 0.2.3 and installs only `libbhc_rts.a` ([#47](https://github.com/arcanist-sh/bhc/issues/47)); use the install script instead |
| crates.io | The release workflow has a publish job, but the `bhc` crate is not on crates.io yet |
| Nix, asdf/mise, AUR, Scoop, WinGet, Docker | Not packaged |

## Platforms

| Platform | Release archive |
|----------|-----------------|
| macOS aarch64 (Apple Silicon) | Yes (`aarch64-apple-darwin`) |
| Linux x86_64 | Yes (`x86_64-unknown-linux-gnu`) |
| macOS x86_64 (Intel) | No |
| Linux aarch64 | No |
| Windows | No |

Other platforms can try a source build. The WebAssembly target
(`--target wasm32-wasi`) is available from either release build.

## What a release archive contains

- `bhc`, the compiler driver
- `lib/libbhc_*.a`: the runtime (`libbhc_rts.a`) and the standard library
  archives that compiled programs link against

The REPL (`bhci`) and the language server (`bhc-lsp`) are not in the archives
yet; build them from source with `cargo build --release -p bhci -p bhc-lsp`.
