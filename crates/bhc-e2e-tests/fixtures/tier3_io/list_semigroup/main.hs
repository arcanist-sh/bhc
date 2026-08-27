module Main where

-- `<>` at a list type had no builtin Semigroup instance, so it lowered to an
-- unresolved-method stub and trapped at runtime. `print (xs ++ ys)` printed
-- an address, because the list-shaped-expression check knew `append` but not
-- the `++` spelling the source actually carries.
main :: IO ()
main = do
  print ([1, 2] <> [3 :: Int])
  print (([] :: [Int]) <> [7])
  putStrLn ("ab" <> "cd")
  print (length ([1, 2] <> [3 :: Int]))
  print ([1, 2] ++ [3 :: Int])
  print (concat [[1], [2, 3 :: Int]])
