-- A second container use of a pattern-bound variable must not undo what the
-- first one learned: `any (== True)` over a `[Int]` is a type error whether
-- or not `null cs` follows it.
g :: (Int, [Int]) -> Bool
g (_, cs) = any (== True) cs && null cs

main :: IO ()
main = print (g (0, [1]))
