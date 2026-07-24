-- foldl' op z (map f xs) with a LAMBDA mapper.
--
-- H26-SPEC 8.1 Pattern 4. Unlike the sum/map rewrite (Pattern 3), this fusion
-- is type-agnostic: `op` (+) and `z` (0) come from the source, so it needs no
-- populated element type and fires even though the mapper `\x -> x * 2` has no
-- known type in Core. Result 30 guards that the fused single strict foldl' is
-- semantics-preserving.
import Data.List (foldl')

main :: IO ()
main = print (foldl' (+) 0 (map (\x -> x * 2) [1, 2, 3, 4, 5 :: Int]))
