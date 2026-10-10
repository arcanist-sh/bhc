# Open repros

Self-contained programs for the gaps the pandoc campaign is currently waiting
on. They need no pandoc sources — `bhc Xxx.hs -o xxx && ./xxx` is enough.

| File | What it shows |
| --- | --- |
| `GND5.hs` | PASSES: a handwritten bind chain and `liftA2 (++)` over a GND newtype both print `ab`. The control for the two below. |
| `GND2.hs` | FAILS (segfault): `instance Semigroup a => Semigroup (F s a) where (<>) = liftA2 (<>)` — pandoc's `Future`, and what `mconcat <$> many1 inline` folds with. Inside an instance method the (eta-expanded) parameters carry no type, so the desugared binds stay the generic builtins instead of the newtype's own. |
| `GND3.hs` | The same, written eta-expanded (`a <> b = liftA2 (<>) a b`), which fails the same way — so it is not about the point-free form. Its first line also shows the `<>` BUILTIN on two untyped Strings prepending a NUL. |

| `POINTFREE_CON.hs` | FAILS (`stub: Outer not implemented`): a class method defined point-free as a partially applied CONSTRUCTOR — `instance Applicative Outer where pure = Outer "o"` — is lowered as a call to a stub named after the constructor. Eta-expanding the method (`pure a = Outer "o" a`) makes the same program print `oi5`/`oi7`. |

Dispatching the GND binds at the ENCLOSING instance type was tried on
2026-10-10 and reverted: it did not fix the target.
