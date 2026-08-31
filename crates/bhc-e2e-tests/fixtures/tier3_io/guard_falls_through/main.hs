-- A guard that FAILS must fall through to the next equation. The guarded RHS
-- was desugared to a nested `if` whose last branch is a pattern-match error,
-- and the equation's guards were dropped from the HIR — so the decision tree,
-- which is built from patterns alone, saw an irrefutable equation and pruned
-- everything after it.
module Main where

size :: Int -> String
size v | v > 100 = "huge"
size v | v > 3 = "big"
size v = "small " ++ show v

both :: Int -> String
both v | v > 3, v < 10 = "middling"
both v = "outside"

-- Guards within ONE equation already chained correctly; keep them working.
sign :: Int -> String
sign v | v > 0 = "pos"
       | v < 0 = "neg"
       | otherwise = "zero"

main :: IO ()
main = do
  putStrLn (size 500)
  putStrLn (size 5)
  putStrLn (size 1)
  putStrLn (both 5)
  putStrLn (both 50)
  putStrLn (sign 1)
  putStrLn (sign (-1))
  putStrLn (sign 0)
