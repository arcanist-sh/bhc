# BHC-RFC-0003 — A Typed Agent Harness

**Document ID:** BHC-RFC-0003
**Status:** Idea / research note. Nothing here is planned work; it records a question and the cheapest way to answer it.
**Scope:** Whether an agent call (a model completing a typed function at runtime) should be a library in BHC, a compiler-known effect, or neither
**Date:** 2026-11-20

---

## 1. Where the question comes from

NVIDIA's NOOA ([NVIDIA-NeMo/labs-OO-Agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents), arXiv 2607.20709) models an agent as a Python class. Methods are capabilities, fields are state, the docstring is the prompt, and the signature is the contract. A method whose body is `...` is completed at runtime by the model, which acts by writing Python against typed services:

```python
class BillingAgent(Agent, llm=llm):
    categories: set[str] = {"food", "travel", "other"}

    async def categorize(self, prices: list[ItemPrice]) -> list[CategoryTotal]:
        """Group items by self.categories; sum prices for each category."""
        ...
```

Matt Staats presented it at Agentic Engineering Days Zurich (2026-11-20). Two of his results matter here:

- **The harness, not the model, moved the numbers.** Same model, same skill: agent-managed Markdown notes reached 38.4% mean RHAE on ARC-AGI-3, NOOA's memory API reached 50.2%.
- **Structure is cheaper than thinking.** Matching the Markdown score cost about $290 in estimated API spend with the structured harness against about $1,040 without it.

(Numbers from the talk's slides; NVIDIA's own write-up is the source to cite.)

NOOA's weak point is the host language. A Python signature is a promise checked when the call happens; NOOA validates the output and retries until it fits. Nothing stops the application and the agent from disagreeing about a type until production.

## 2. The thesis

BHC's mission is "predictability over folklore". An agent call is the least predictable thing a program can do, so it is exactly where a compiler that checks everything it can should have the most to say. The contract between deterministic code and a delegated judgement should be a **type the compiler checks**, and the costs of the delegation (retries, budget, latency, memory) should be **visible in that type** instead of hidden in a runtime loop.

## 3. Three ways to do it

| Option | What it is | What the compiler checks | Cost to BHC |
|---|---|---|---|
| A. Library | an `Agent` effect in a stdlib package (`bhc-agent`), built on `IO`, with a typeclass that turns a type into a JSON schema and a parser | the Haskell types at the call site; decoding of the model's answer is a runtime parse | none in the compiler |
| B. Library plus elaboration | as A, but `deriving (Delegate)` generates schema, parser and prompt skeleton from the type and its Haddock comment | as A, plus that every delegated type has a total decoder | a deriving strategy |
| C. Compiler-known effect | `delegate` is a primitive the compiler knows, like `IO` in a profile; budgets and retries are part of its type | as B, plus static budget composition and a ban on delegation in profiles that forbid it (`realtime`, `embedded`) | type checker and profile work |

A sketch of what B could look like at the use site:

```haskell
data CategoryTotal = CategoryTotal { category :: Category, total :: Double }
  deriving (Generic, Delegate)

-- | Group items by category; sum the prices for each category.
categorize :: [ItemPrice] -> Agent '[Retries 3, Budget 0.02] [CategoryTotal]
categorize = delegate
```

The type says what must come back, how often the model may try, and what it may cost. The Haddock comment is the prompt. Composition of two delegated calls sums their budgets in the type.

## 4. What BHC brings that Python cannot

- **Totality of the boundary.** A delegated result is parsed into a closed type or the call fails with a typed error; there is no partially valid value.
- **Profiles.** A `realtime` or `embedded` build can reject any program that contains a delegation, the same way profiles already constrain allocation and concurrency.
- **Structured concurrency.** Delegated calls inherit cancellation and deadlines from the server profile's scheduler (see `docs/profiles.md`), so a slow model is a deadline, not a hang.
- **The commons.** If RFC-0001 lands, a delegated function with its contract, prompt and evaluation record could be a manifest like any other, and "this delegation has been verified on these cases" becomes something the registry records.

## 5. What stays hard

- **Types do not make answers right.** The compiler can guarantee the shape of a result, not its truth. Evaluation remains a runtime and data problem; the type only fixes what is evaluated.
- **Type-level budgets** are easy to sketch and hard to make ergonomic; they may belong in a runtime policy instead.
- **Code as action.** NOOA's model writes Python and runs it. A Haskell harness would need a safe action surface (an interpreter for a small typed DSL, or generated Haskell compiled at runtime), and both are large.
- **Scope.** BHC's road to 1.0 is Pandoc and the stdlib. This must not compete with that.

## 6. Recommendation and a de-risking experiment

**Recommendation:** start with option A as an ordinary package, outside the compiler. Move to B only if the deriving is clearly where the value is, and to C only if a real program shows that static budgets or profile bans catch something a library cannot.

**Experiment (small, after 1.0):** port NOOA's `BillingAgent` and one memory-heavy example to a `bhc-agent` library prototype against a local model. Measure: (1) how many decoding failures the typed boundary turns into compile-time or first-call errors, (2) lines of harness code against the Python version, (3) whether the Haddock-as-prompt convention holds up on more than toy examples. If (1) is near zero, the compiler has little to add and A is enough.

## References

- NVIDIA, [NVIDIA-NeMo/labs-OO-Agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents) and [Six Agent Harness Capabilities for Higher Model Performance](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/)
- Xingyao Wang et al., [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030), ICML 2024
- Raffael Schneider, [Harness types in the compiler](https://raskell.io/notes/harness-types-in-the-compiler/), the note this RFC grew from
