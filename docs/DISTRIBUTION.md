# bhc distribution channels

Where bhc is published, and what each channel installs. Current release:
v0.2.22.

## Recommended

| Channel | Install | Notes |
|---------|---------|-------|
| Install script | `curl -fsSL https://arcanist.sh/bhc/install.sh \| sh` | Installs to `~/.bhc`; set `PATH=~/.bhc/bin` and `LIBRARY_PATH=~/.bhc/lib` |
| hx | `hx toolchain install --bhc 0.2.22` | Then `hx build --backend bhc`, or `[compiler] backend = "bhc"` in `hx.toml` |
| GitHub Releases | [releases](https://github.com/arcanist-sh/bhc/releases) | `.tar.gz` per platform with `bhc`, `bhci`, `bhi` and `lib/libbhc_*.a`, plus `SHA256SUMS.txt` |
| From source | `cargo build --release -p bhc -p bhc-rts ...` | See [Getting Started](getting-started.md#from-source); needs LLVM 21 |

The install script checks the archive against the release's `SHA256SUMS.txt`
before unpacking it. To check a download by hand, run
`shasum -a 256 -c --ignore-missing SHA256SUMS.txt` (or `sha256sum -c
--ignore-missing SHA256SUMS.txt`) in its directory; releases before v0.2.30
list the archives under a directory prefix, which `-c` does not match.

## Other channels

| Channel | State |
|---------|-------|
| Homebrew (`arcanist-sh/tap/bhc`) | `brew install arcanist-sh/tap/bhc`; installs `bhc`, `bhci`, `bhi` and every `libbhc_*.a`. The release workflow updates the formula (`scripts/homebrew-formula.sh`); set `LIBRARY_PATH` as the caveats say |
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
- `bhci`, the REPL that `bhc repl` starts (from v0.2.30)
- `bhi`, the IR and kernel-report inspector (from v0.2.30)
- `lib/libbhc_*.a`: the runtime (`libbhc_rts.a`) and the standard library
  archives that compiled programs link against

The language server (`bhc-lsp`) is not in the archives yet; build it from
source with `cargo build --release -p bhc-lsp`.
