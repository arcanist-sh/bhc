-- A do block must end in an expression: one that ends in a `let` (or a
-- `x <- e`) has no result. It compiled, and failed where the block ended.
main :: IO ()
main = do
  print (1 :: Int)
  let v = 2 :: Int
