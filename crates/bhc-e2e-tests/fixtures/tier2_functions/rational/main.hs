module Main where

main :: IO ()
main = do
  -- Construction and normalization
  putStrLn (show (1 % 2))
  putStrLn (show (2 % 4))
  -- Arithmetic
  putStrLn (show (1 % 3 + 1 % 6))
  putStrLn (show (1 % 2 - 1 % 3))
  putStrLn (show (2 % 3 * 3 % 4))
  putStrLn (show (1 % 2 / 1 % 3))
  -- Unary operations
  putStrLn (show (negate (1 % 3)))
  putStrLn (show (abs (negate (2 % 5))))
  putStrLn (show (signum (3 % 7)))
  -- Equality (the key test)
  putStrLn (show (1 % 3 + 1 % 6 == 1 % 2))
  -- Accessors
  putStrLn (show (numerator (2 % 4)))
  putStrLn (show (denominator (2 % 4)))
