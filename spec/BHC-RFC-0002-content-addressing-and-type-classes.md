# BHC-RFC-0002 — Content-Addressing and Type Classes

**Document ID:** BHC-RFC-0002
**Status:** Draft / research note (decides feasibility of [BHC-RFC-0001](BHC-RFC-0001-the-commons.md))
**Scope:** Whether and how a content-addressed commons can identify *real Haskell* definitions in the presence of type classes, coherence, and orphan instances
**Date:** 2026-06-19

---

## 1. Why this is the crux

Content-addressing wants **locality**: a definition's identity should be a function of its content plus the identities of its dependencies, and nothing else. Type classes break locality.

Consider:

```haskell
sort :: Ord a => [a] -> [a]
```

The *meaning* of `sort` at runtime depends on which `Ord` dictionary flows in at the call site. That dictionary is not in the text of `sort`. It is chosen later, globally, by instance resolution over the whole import graph. So the surface definition of `sort` does not, by itself, determine its behavior — and a content hash of the surface definition would be addressing something whose meaning is not yet fixed.

This is precisely why **Unison has no type classes.** Unison passes explicit records (its analogue of dictionaries) by hand, so every definition is genuinely local. BHC does not have that luxury: it compiles real Haskell, and real Haskell's ergonomics *are* type classes. RFC-0001 is only feasible if we can give a class-using definition a well-defined identity. This note argues we can, and says what stays hard.

## 2. The key move: hash the dictionary-passed Core

BHC already eliminates type classes during elaboration. Type checking resolves each constraint to a concrete **dictionary**, and class methods become **field selections** on dictionaries passed as ordinary arguments (`bhc-typeck` implements dictionary passing; the optimizer's dictionary specialization, O.4, operates on exactly this representation). After that pass, `sort` is no longer polymorphic-over-an-implicit-instance; it is a function that takes its `Ord` dictionary explicitly:

```
-- Conceptually, post-elaboration:
sort :: OrdDict a -> [a] -> [a]
sort $dOrd xs = ... ($sel_compare $dOrd) ...
```

> **Status (audited 2026-06-19):** the dictionary-passed Core form is real and already produced. `bhc-hir-to-core/src/dictionary.rs` elaborates instances into dictionary values bound as `$d…` variables, passes them as ordinary `App` arguments, and selects methods via `$sel_N` field selectors; `bhc-core/src/specialize.rs` already optimizes exactly this representation (it rewrites `App($sel_N, $d)` to the Nth field of a known dictionary tuple). What does **not** yet exist is the canonical-binder normalization the hash needs — BHC's Core is name-based (`Symbol` + `VarId`), see §8. So the representation we want to hash exists; making it *stably* hashable is the open work.

**We hash this.** The identity of a manifest is computed over the *dictionary-passed, normalized* Core, not the surface definition. The consequences are exactly what we want:

1. **Identity becomes local again.** Every instance the definition uses is now an explicit dependency — a dictionary value, itself a manifest, with its own content address. `sort`'s identity transitively includes the identity of the `Ord` instance(s) it was elaborated against, the same way it includes any other dependency.
2. **Coherence becomes explicit and addressable.** "Which `Ord Int` did this use?" stops being an ambient property of the import graph and becomes a hash in the dependency list. Two builds that resolved `Ord Int` differently produce two different manifests — correctly, because they *mean* different things.
3. **It composes with the contract axis.** Profile/edition/extensions already parameterize elaboration (RFC-0001 §4.2); instance resolution is just another part of elaboration that the contract pins down.

The price: identity is **coarser but honest**. A generic `sort` and its monomorphic specialization at `Ord Int` are different manifests. You do not get "one `sort` for all instances." You get one manifest per (definition, resolved-instances, contract). That is the right answer — they are different behaviors — and it is the only answer that keeps addressing sound.

## 3. Worked sketch

```haskell
-- Surface
class Eq a => Ord a where compare :: a -> a -> Ordering
instance Ord Int where compare = compareInt

minimumBy :: Ord a => [a] -> a
minimumBy = foldr1 (\x y -> if compare x y == LT then x else y)
```

After elaboration, `minimumBy` is a function of an explicit `OrdDict a`. Its manifest:

```
id            = hash( coreOf(minimumBy_dictpassed) ⊗ contract )
dependencies  = [ id(foldr1), id(compareViaDict), id($sel_compare), ... ]
```

At a use site `minimumBy [3,1,2 :: Int]`, elaboration applies the `Ord Int` dictionary. The *specialized* call (post O.4 dictionary specialization) may inline `compareInt` directly — but specialization is an **optimization that must preserve the manifest's identity**, not change it. The canonical identity is the pre-specialization dictionary-passed form; specialization affects the compiled *artifact* (which is keyed by contract anyway), not the manifest's address. This separation matters: it keeps identity stable across optimization-level changes.

## 4. Orphan instances

An orphan instance (declared in neither the class's module nor the type's module) is the classic poison for any identity-by-content scheme, because it makes the *meaning* of a definition depend on whether some unrelated module happened to be imported.

Under dictionary-passed hashing this is *contained but not free*:

- Because every instance used is captured in the dependency closure by hash, a manifest that relies on an orphan instance **captures that orphan explicitly** — its identity includes the orphan's hash. The orphan stops being ambient and becomes addressed. Good.
- The danger that remains is *incoherence*: two manifests in one link closure that captured *different* orphan instances for the same `(class, type)` pair. This is the genuine hard case and is handled at link time (§5), not at identity time.
- **Recommendation:** BHC should diagnose reliance on orphan instances when producing a manifest (a warning today; a policy knob for the commons later), and record captured instances in the evidence bundle so consumers can see exactly which incoherence risks a manifest carries.

## 5. Coherence at commons scale

Identity makes each manifest's instance choices explicit; it does not by itself guarantee that a *closure* of manifests agrees. The link-time obligation:

> For every `(class, type)` pair used in a compiled closure, all manifests must have been elaborated against the *same* instance.

BHC already performs instance resolution for a single compilation; the commons extends this to a check over the manifest closure. Failure is a coherence error, surfaced like any other (RULE-002 territory: fail at compile time, not runtime). Overlapping, flexible, and undecidable instances widen the space of closures that *can* be incoherent and should be flagged as elevated-risk in the evidence bundle.

## 6. Adjacent elaboration features

The same "hash after elaboration" principle resolves the neighbors of the class problem:

- **Superclasses** — dictionary construction already materializes superclass edges (e.g. `Ord` carrying its `Eq`); hashing the dictionary value captures them.
- **`deriving` (stock, newtype, anyclass)** — deriving expands to instances/coercions during elaboration; hash the expansion, not the `deriving` clause.
- **Type families / associated types** — reduction happens in elaboration (BHC implements open/closed/associated families); hash the *reduced* types.
- **Defaulting** — numeric defaulting is resolved in elaboration; the resolved type is what gets hashed (and is pinned by the edition in the contract).

## 7. What stays genuinely hard

- **Semantic vs. syntactic identity** (inherited from RFC-0001 §4.4): dictionary-passed hashing still only unifies definitions that *elaborate the same way*. It will not discover that two different `Ord` instances induce the same ordering.
- **Higher-rank / impredicative dictionaries** and constraint-kind machinery (`ConstraintKinds`, quantified constraints) push more into elaboration and need their normalized Core forms specified before hashing is trustworthy there.
- **`INCOHERENT`/`OVERLAPPING` pragmas** intentionally make resolution nondeterministic-ish; manifests using them should be marked and likely excluded from the shared (networked) commons even if allowed locally.
- **Cross-version instance drift**: if a dependency's instance changes, every dependent manifest's identity changes (by design). This is correct but means instance changes ripple — the commons must make that ripple cheap to recompute (which is exactly what a persisted `bhc-query` store is for).

## 8. Recommendation and a de-risking experiment

**Recommendation:** Adopt dictionary-passed, de-Bruijn-normalized Core as the canonical hashing representation. It restores locality, makes coherence explicit, composes with the contract axis, and reuses elaboration BHC already performs. Treat orphan/overlapping/incoherent instances as evidence-bundle risk flags and as the responsibility of the link-time coherence check, not the identity function.

**Go/no-go experiment (small, decisive):** take ~10 `base`-style functions with class constraints (`sort`, `nub`, `maximum`, `lookup`, a `Show` and a `Num` user).

**Step 0 (prerequisite — confirmed necessary by the source audit):** implement canonical-binder normalization of dictionary-passed Core. BHC's Core is name-based today (`Symbol` + `VarId`), so without this step the soundness check below fails trivially on differing local names. This is the *only* piece of new compiler work the experiment depends on; everything after it is measurement.

Then, for each function:

1. Elaborate to dictionary-passed Core; normalize; compute `hash(meaning ⊗ contract)`.
2. **Soundness:** reimplement each in a different surface style (rename locals, reorder `where`-binds); confirm identical hash.
3. **Sensitivity:** swap a relied-upon instance (e.g. a flipped `Ord`); confirm the hash changes.
4. **Coherence:** construct a two-manifest closure with conflicting orphan `Ord Int`; confirm the link-time check rejects it.

If (2)–(4) hold on this sample, content-addressing real Haskell is buildable and RFC-0001 can proceed. If (2) fails *after* Step 0, the normalization of dictionary-passed Core is underspecified and must be fixed before anything else.
