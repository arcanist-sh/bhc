# BHC-BRIEF-0001 — Commons, Stage 1: a persistent content-addressed Core store

**Document ID:** BHC-BRIEF-0001
**Status:** Ready for implementation
**Owner:** build agent
**References:** [BHC-RFC-0001](BHC-RFC-0001-the-commons.md) §4, §11; [BHC-RFC-0002](BHC-RFC-0002-content-addressing-and-type-classes.md) §8
**Audited against source:** 2026-06-19
**Date:** 2026-06-19

---

## Goal

Make BHC **never recompile the same meaning twice**: persist the query cache as an on-disk, content-addressed store keyed by `hash(meaning ⊗ contract)`. This is RFC-0001 stages 1–2, and it is the foundation everything else in the commons sits on.

It ships value on its own — a persistent cross-invocation incremental cache — with **no network, no signing, no governance**. If the commons program stops here, BHC still gained a real speedup. That independence is the point of doing this stage first.

## The gate: Task 0 (go / no-go)

Before building any store, prove that real Haskell can be content-addressed at all. This is RFC-0002 §8, and the audit confirmed it is the one piece of genuinely new compiler work.

**Task 0 — canonical-binder normalization + hashing experiment.**

- Implement a normalization pass over **dictionary-passed Core** (the form already produced by `bhc-hir-to-core/src/dictionary.rs`): convert name-based binders (`Symbol` + `VarId`, `bhc-core/src/lib.rs`) to de Bruijn indices or a deterministic canonical renaming.
- On ~10 `base`-style functions with class constraints: confirm **soundness** (different surface styles → identical hash), **sensitivity** (swap a relied-upon instance → hash changes), and **coherence** (a closure with conflicting orphan `Ord Int` is rejected).
- **Exit:** if soundness holds after normalization, proceed. If it fails, STOP and revise RFC-0002 before writing any store code.

## Tasks (in order)

1. **Canonical-binder normalization** *(new pass, `bhc-core`)*
   BHC's Core `Var` is name-based today (`bhc-core/src/lib.rs`). Add a normalization producing a canonical form over dictionary-passed Core.
   *Acceptance:* alpha-equivalent definitions normalize identically; `bhc-core/src/specialize.rs` output is unaffected (normalization is identity-preserving for already-canonical input).

2. **Cryptographic content address** *(new, `bhc-core`/`bhc-interface`)*
   Define `Address = digest(normalized_core ⊗ contract)`, where `contract = { profile, edition, enabled extensions, sorted dependency Addresses }`. Use **SHA-256** (hx already depends on `sha2`) or **BLAKE3** if hashing throughput matters. Do **not** reuse `bhc-query`'s `FxHash` for addresses — that stays for in-process memoization only.
   *Acceptance:* the same source under the same contract yields the same `Address` across runs and machines; the contract encoding is documented and stable.

3. **Real interface hashing** *(`bhc-interface`)*
   Replace `compute_module_hash` (`bhc-interface/src/generate.rs`, today hashes only the module *name*) with a content hash, and populate `InterfaceDependency.hash` (today written as `0`).
   *Acceptance:* changing a dependency's content changes the `Address` of every dependent.

4. **Persist the query store** *(`bhc-query`)*
   Back `bhc-query` (`QueryId { key_hash }`, `DashMap`) with an on-disk store keyed by `Address`: load on startup, write on completion, integrate with hx cache-dir conventions.
   *Acceptance:* a second `bhc`/`hx build` of unchanged meaning is a measured cache hit with no codegen; reuse survives across process invocations.

5. **(stretch) Borrow-by-address** *(seed for stage 2)*
   Resolve a dependency by `Address` from the local store instead of recompiling it.
   *Acceptance:* a dependency present in the store is linked without re-elaboration.

## Out of scope (later stages, do not build here)

- Attestations / signing (stage 3 — needs a generic `hx-crypto` module + a signing path; verification already exists).
- Networked registry, names-as-metadata-over-hashes (stage 5).
- MCP `commons.lookup/verify/contribute` tools (stage 4 — `hx mcp` tool registration is in `hx-cli/src/commands/mcp.rs`).
- Coherence governance / web-of-trust policy.

## Known hazards (RFC-0001 §12, RFC-0002 §4–§7)

- **Impurity:** Template Haskell, CPP, FFI, and `IO` resist addressing/determinism. Start with the pure, total core; fence or exclude impure definitions from the store initially.
- **Profile combinatorics:** the `Address` includes the contract, so N definitions × 6 profiles multiply the store. Materialize per-contract lazily.
- **Two hashes, kept apart:** non-cryptographic `FxHash` for in-process memoization; cryptographic digest for `Address`. Never let the memo hash leak into an address.
- **Instances:** orphan/overlapping instances are captured in the dependency closure by hash; flag elevated-risk ones in diagnostics (RFC-0002 §4–§5).

## Success metric

- **Reproducible:** two machines compiling the same source under the same contract derive the same `Address` for every top-level definition.
- **Fast:** a warm-store rebuild of an unchanged module is a cache hit with zero codegen.
