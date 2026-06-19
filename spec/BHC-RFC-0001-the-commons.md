# BHC-RFC-0001 — The Commons

**Document ID:** BHC-RFC-0001
**Status:** Draft (for discussion)
**Scope:** A content-addressed commons of attested Haskell definitions, and BHC's role compiling them into compute artifacts
**Related:** [BHC-RFC-0002 — Content-Addressing and Type Classes](BHC-RFC-0002-content-addressing-and-type-classes.md), `bhc-query`, `bhc-interface`, `arcanist.sh/manifesto`, `arcanist.sh/commons`
**Date:** 2026-06-19

---

## 1. Summary

This RFC proposes a system in which the unit of reuse in the BHC/hx ecosystem is not a package but a **manifest**: a single, content-addressed Haskell definition that carries (a) an identity derived from its elaborated meaning under a stated compilation contract, (b) an **evidence bundle** recording exactly what has been checked about it, and (c) one or more signed **attestations** recording who vouched for it and how.

Manifests live in **the commons**: a content-addressed store where names are metadata over hashes, identical definitions are stored once, and verification accumulates rather than being repeated. BHC compiles a chosen *closure* of manifests, under a chosen runtime profile, into a reproducible compute artifact — which is itself a manifest.

The thesis in one line: **a query-based incremental compiler is already a content-addressed cache; the commons is that cache made persistent, shareable, and attested.**

## 2. Motivation

The manifesto's pivot is *when generation is cheap, verification is the bottleneck*. The corollary we have not yet acted on: **if verification is the expensive step, repeating it is the waste.** Today verification is per-program and ephemeral — every agent re-derives, re-checks, and discards. A commons makes verification a durable, citable, shared result.

This is also a strategic position. On agentlanguages.dev, BHC/hx is filed under *"Orchestration + Syntactic — not a new language."* The commons annexes the **Verification** camp's territory (cf. Aver exporting to Lean/Dafny, Boruna's "hash-chained evidence bundles") *without* becoming a new language. The differentiator: the others invented languages so agents could be verified; we build the commons and the meaning engine for a language that already has the proof machinery.

The unique substrate triangulation: **Unison's content-addressing × Haskell's types-as-proofs × BHC's runtime profiles and compute backends.** Unison has the addressing but no performance story; Erlang/Elixir can address code but carry no meaningful evidence (untyped); BHC is the one place all three meet, and the commons therefore produces *fast, vouched-for compute artifacts*, not just elegant references.

## 3. Terminology

Defined canonically in the [glossary](https://arcanist.sh/glossary/). In brief:

- **Manifest** — a content-addressed Haskell definition carrying its evidence and provenance; the semantic artifact, completed.
- **Attestation** — a signed, machine-checkable claim about a manifest (reproducible build, property suite passed, reviewed-by, vouched-by).
- **The commons** — the shared content-addressed namespace of manifests and attestations; names are metadata over hashes.
- **Manifest identity** — `hash(meaning ⊗ contract)`; see §4.

We deliberately do **not** use "truth manifest." A well-typed function proves consistency with a specification, not fitness for intent (`head :: [a] -> a` is well-typed and partial). "Truth" overclaims and contradicts the manifesto's "report what it cannot prove." The unit is a *manifest carrying attestations*.

## 4. Identity model: `hash(meaning ⊗ contract)`

A manifest's address is **not** a hash of source text. It is a hash of two things together:

### 4.1 Meaning

The hash is computed over the **elaborated** definition — post-typecheck, post-desugar Core IR (`bhc-core`) — with all surface-level incidentals (comments, layout, formatting, choice of local names) erased, so that two definitions that elaborate to the same normalized Core have the same identity. This is the Unison move, applied to BHC's Core.

> **Status (audited 2026-06-19):** BHC's Core IR is today *name-based* — binders are interned `Symbol` + `VarId(u32)` (`bhc-core/src/lib.rs`), with no alpha-normalization pass. A canonical-binder normalization (de Bruijn indexing, or deterministic canonical renaming) is therefore a **prerequisite** for content-addressing — without it, two definitions that differ only in local variable names hash differently. This is task zero of the stage-1 brief (BHC-BRIEF-0001); the rest of this section assumes that pass exists.

Hashing *after* elaboration (not surface AST) is essential, because elaboration is where the ambiguities that would otherwise break addressing are resolved — most importantly, type-class instance selection (see §4.4 and RFC-0002).

### 4.2 Contract

BHC's defining property is that the same source has different meanings under different runtime profiles. Identity must therefore fold in the **contract** the definition was elaborated against:

| Contract component | Why it is in the identity |
|---|---|
| Runtime profile (`default`/`server`/`numeric`/`edge`/`realtime`/`embedded`) | Same source, genuinely different behavior and artifact |
| Haskell edition (`Haskell2010`/`GHC2021`/`GHC2024`/`H26`) | Changes defaults and semantics |
| Enabled language extensions | Can change elaboration |
| Dependency manifest hashes | Meaning depends on what it was elaborated against |

The same definition under two profiles is **two manifests**, because it is two behaviors. This is not a defect; it is the "semantic artifact" thesis made literal: *the unit is meaning under a stated contract, and that pair is its name.*

### 4.3 Relationship to existing machinery

The identity function is largely already implicit in `bhc-query` (the incremental/query compiler) and `bhc-interface`/`.bhi` (separate compilation). Query systems are content-addressed by construction: they hash inputs and memoize results, and `bhc-query` already does exactly this (`QueryId { key_hash }`, `FxHasher`). The work is to (a) make the key explicitly `hash(meaning ⊗ contract)`, (b) persist the store, and (c) make it shareable.

Two corrections from the source audit (2026-06-19):

- **Cryptographic hash required.** `bhc-query` keys on a *non-cryptographic* 64-bit `FxHash` — correct for in-process memoization, unsafe as a commons address, because attestations bind to the address and it must be collision-resistant. Commons identities must use a cryptographic digest (SHA-256 or BLAKE3). `bhc-query` proves the *architecture*, not the hash function; keep the two hashes distinct.
- **The slot already exists.** `.bhi` interfaces already carry a `module_hash: u64` field and an `InterfaceDependency.hash` plumbing path (`bhc-interface`), but both are placeholders — `compute_module_hash` (`bhc-interface/src/generate.rs`) hashes only the module *name*, and dependency hashes are written as `0`. Filling these with real content hashes is the most concrete first PR. `.bhi` is the natural on-disk carrier for a manifest's public face.

### 4.4 What identity is *not*

Identity is **normalized syntactic** equivalence, not semantic equivalence. Two extensionally-equal implementations of `sort` will, in general, hash differently. True semantic dedup is undecidable; the commons dedups definitions people *write the same way* (after elaboration), not all definitions that *mean the same thing*. Unison has the identical limitation. We state it plainly rather than imply more.

## 5. The manifest

A manifest is a record. Sketch (wire format TBD — CBOR/JSON; field names illustrative):

```
Manifest
  id            : Hash              -- hash(meaning ⊗ contract); the address
  core          : CoreDef          -- elaborated, de-Bruijn-normalized Core
  type          : Type             -- inferred/checked type
  contract      : Contract         -- profile, edition, extensions
  dependencies  : [Hash]           -- transitive closure, by manifest id
  evidence      : EvidenceBundle   -- §6
  attestations  : [Attestation]    -- §7 (detachable; may be fetched separately)
  names         : [QualifiedName]  -- metadata only; never part of id
```

Names are stored *alongside* the manifest but are never inputs to `id`. A name resolves to a hash; many names may alias one hash; a rename mutates the name index, not the manifest.

## 6. The evidence bundle

The evidence bundle is "report what it cannot prove," rendered as data. Every claim is tagged with its **strength**:

| Strength | Meaning |
|---|---|
| `Established` | Mechanically checked by the compiler (types always; totality/exhaustiveness when proven; discharged refinements) |
| `Tested` | Checked by execution over cases (property tests, with the property statement and seed/coverage attached) |
| `Asserted` | Claimed by an author, not checked |

Categories BHC can populate today or near-term:

- **Type consistency** — `Established`, always (typeck).
- **Totality / exhaustiveness** — `Established` when the pattern-match compiler (decision trees + exhaustiveness checking, already implemented) reports no fallthrough and no partial primitives are reachable.
- **Properties** — `Tested`; statements + results from `hx test` / property suites, attached with enough metadata to re-run.
- **Refinements / contracts** — `Established` when discharged to an SMT backend (a future `verify`-block facility, cf. Aver/Vera/Intent on agentlanguages.dev); otherwise `Asserted`.
- **Performance contract** — kernel/fusion reports (already emitted in the numeric profile) recorded against the artifact.

The bundle must never launder an `Asserted` claim into apparent authority. A consumer's policy (see §7.3) decides what minimum strength it will build on.

## 7. Attestations

### 7.1 What is signed

An attestation is an Ed25519 signature over a statement binding (the verification primitive already exists in hx — see the status note below):

```
Attestation
  manifest      : Hash             -- the manifest id
  evidence      : Hash             -- digest of the evidence bundle
  artifact      : Maybe Hash       -- reproducible compiled-artifact hash, if any
  claim         : Claim            -- ReproducibleBuild | PropertiesPass | ReviewedBy | VouchedBy | ...
  signer        : PublicKey
  signature     : Sig
```

> **Status (audited 2026-06-19):** Ed25519 *verification* is implemented and reusable — `hx-solver/src/bhc_platform.rs::verify_ed25519`, `ed25519-dalek` v2, with `HX_BHC_PLATFORM_PUBKEY` key pinning (and a deliberate "don't trust a registry-supplied key" warning). Net-new work: (1) a *signing* path (signing currently exists only in tests), and (2) lifting verification out of the snapshot-specific `SnapshotError` into a generic `hx-crypto`-style module that signs/verifies a digest against a pinned key. SHA-256 fingerprinting already exists for reproducibility (`hx-solver/src/cache.rs::compute_deps_fingerprint`).

### 7.2 Reproducibility is the objective floor

`ReproducibleBuild` is the load-bearing attestation because no one has to be *trusted* for it: any party can recompile the manifest under the stated contract and confirm they reach the same `id` and the same `artifact` hash. This requires deterministic compilation — see §12 on GC/runtime determinism. Everything social sits on top of this objective floor.

### 7.3 Policy-gated trust

Above reproducibility, trust is a policy decision: which signers' reviews are required, which vouches are accepted, which organizations are excluded. This is conceptually `zentinel-agent-policy` applied to definitions — a declarative policy the resolver enforces at link time. Consumers choose; the commons enforces.

### 7.4 Transparency

Attestations are published to an append-only log (Certificate Transparency / Sigstore lineage) so that "who vouched for what" is auditable and non-repudiable. The log is not a trust root; it is a record.

## 8. The commons

- **Store** — content-addressed: `id → Manifest`, plus a detachable `id → [Attestation]` index and a mutable `Name → id` index.
- **Borrow by hash** — depending on a manifest by `id` guarantees identical meaning under the identical contract. The class of dependency conflicts caused by name collisions disappears (Unison's result).
- **Dedup + merge** — independently-authored identical definitions collapse to one `id`; their evidence and attestation sets pool. Convergence strengthens an object rather than duplicating it.
- **Local first, networked later** — a per-machine/per-org commons (just a persisted `bhc-query` store) delivers the no-rebuild and dedup wins with no network and no governance. The shared registry is the last stage, because it is the part that needs a trust model proven first.

## 9. Compute artifacts

BHC compiles a **closure** of manifests — a root set plus transitive dependencies, all by `id` — under one profile, into a binary / WASM module / GPU kernel. Two requirements:

1. **Coherence check at link time.** The closure must not contain conflicting type-class instances for the same type (the incoherence problem, now at commons scale). BHC already resolves instances; this extends that check to the manifest closure. See RFC-0002.
2. **The artifact is a manifest.** It is content-addressed, reproducible, and attestable. The output of the commons is more commons.

## 10. Integration with existing BHC/hx

| Commons concept | Existing machinery to evolve |
|---|---|
| `hash(meaning ⊗ contract)` identity | `bhc-query` (incremental, already input-hashed), `bhc-core` (Core IR) |
| Manifest public face / borrow-by-hash | `bhc-interface`, `.bhi` files, separate compilation (`-c`, `--package-db`) |
| Evidence: totality | Pattern-match compiler + exhaustiveness checking (implemented) |
| Evidence: properties/perf | `hx test`, `hx coverage`, numeric kernel reports |
| Attestations (Ed25519) | `HX_BHC_PLATFORM_PUBKEY` snapshot verification, deterministic lockfiles |
| Name resolution / fetch | `hx add` / `hx info` (Hackage resolver) generalized to commons hashes |
| Agent access | `hx mcp` — new tools `commons.lookup`, `commons.verify`, `commons.contribute` |

The agent loop, concretely: via `hx mcp`, an agent requests *a total `parseRFC3339 :: Text -> Either Error UTCTime` satisfying property `roundtrips`*; the commons returns a manifest + evidence + attestations; the agent composes it. If none matches, the agent generates one, BHC verifies it, and — if reproducible and policy-passing — it is contributed back. Verification becomes cumulative.

## 11. Staged rollout

Each stage is independently useful even if the program stops there.

1. **Persist `bhc-query`.** On-disk content-addressed store keyed by `hash(meaning ⊗ contract)`. Pure speedup: "BHC never recompiles the same meaning twice."
2. **Local commons.** Address and borrow `.bhi` interfaces by hash within a machine/org. Delivers dedup + no-rebuild, no network.
3. **Attestations on what exists.** Sign `(manifest id, evidence digest, reproducible artifact hash)` with the existing Ed25519 machinery. "Vouched" becomes machine-checkable.
4. **MCP surface.** `commons.lookup/verify/contribute`. The smallest demo with the biggest payoff: agents cite instead of regenerate.
5. **Shared registry + names-over-hashes + governance.** The networked commons, last.

## 12. Open questions

- **Type classes & coherence.** The central hard problem; addressed in [RFC-0002](BHC-RFC-0002-content-addressing-and-type-classes.md). Resolution: hash the *dictionary-passed* Core.
- **Syntactic ≠ semantic identity (§4.4).** Accept coarse-but-honest dedup; do not imply more.
- **Effect boundaries.** `IO`, FFI, Template Haskell, and CPP resist addressing/determinism. The commons is cleanest over the pure, total core; effectful edges need explicit boundaries (cf. Unison abilities) and may be gated the way `hx plugins trust` gates untrusted plugins.
- **Reproducible artifacts vs. the runtime.** A `ReproducibleBuild` attestation requires deterministic codegen *and* a runtime whose observable identity is stable. Profiles with nondeterministic parallel reductions (e.g. parallel float sums) must either be excluded from artifact-level reproducibility or attested only at the manifest (pre-codegen) level. Decide the granularity per profile.
- **Profile combinatorics.** `N` manifests × `6` profiles is a real multiplier on the store. Lazy/on-demand materialization per contract.
- **Governance / Sybil.** Who may vouch? Reproducibility is objective; social vouching needs a policy model and likely staking/reputation. Out of scope for stages 1–4.

## 13. Prior art

- **Unison** — content-addressed code; names as metadata; no builds; the direct inspiration.
- **Nix** — reproducible, content-addressed builds; `.bhi`/lockfiles are already Nix-shaped.
- **Proof-carrying code** (Necula) — artifacts that carry the evidence of their own safety.
- **Sigstore / SLSA / in-toto** — attestations for software supply chains; the provenance layer is, in effect, *Sigstore for definitions*.
- **Certificate Transparency** — append-only public logs as auditable record, not trust root.
- **agentlanguages.dev Verification camp** — Aver (exports to Lean 4/Dafny), Vera/Prove (contracts, Z3), Boruna (hash-chained evidence bundles), Tacit (BLAKE3-addressed definitions). The commons composes their ideas for real Haskell.

The novelty is the **synthesis**, plus the profile-aware `meaning ⊗ contract` identity, plus the fact that the output is fast compute and not only elegant references.

## 14. Non-goals

- Not a new language. Real Haskell, elaborated by BHC.
- Not a claim that manifests are "correct" or "true." They are *checkable*, with the evidence attached.
- Not a replacement for Hackage in stage 1; a content-addressed layer that can coexist with and eventually subsume name-based distribution.
