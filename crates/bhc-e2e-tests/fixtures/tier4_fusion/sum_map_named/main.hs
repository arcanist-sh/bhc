-- sum (map f xs) where f is a NAMED function f :: Int -> Int.
--
-- This is the `sum/map -> foldl'` guaranteed-fusion pattern (H26-SPEC 8.1
-- Pattern 3). The rewrite gates on the mapped function's codomain being Int,
-- read from `f.ty()` in the Core simplifier. That only works because the typed
-- Core IR work (spec/BHC-BRIEF-0002) populates `f`'s Core `Var` type during
-- lowering; before it, `f.ty()` was `Fun(Error, Error)` and the rewrite never
-- fired on a real program. The correct result here (30) guards that the fused
-- strict-foldl' form is semantics-preserving.
dbl :: Int -> Int
dbl x = x * 2

main :: IO ()
main = print (sum (map dbl [1..5]))
