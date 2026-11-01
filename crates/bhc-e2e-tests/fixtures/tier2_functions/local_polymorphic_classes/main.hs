-- Local (let/where) functions using builtin class methods, used at several
-- types: each use gets a copy at its type (#57).
module Main where

label :: Show a => a -> String
label a = show a ++ "!"

main :: IO ()
main = do
  print (let f x = x * x in map f [1, 2 :: Integer])
  let bigger a b = if a > b then a else b
  print (bigger (2 ^ 70) (5 :: Integer), bigger "ab" "b")
  let inc x = x + 1
  print (inc (2 ^ 64 :: Integer), inc (1.5 :: Double), inc (3 :: Int))
  let same a = a == a
  print (same (7 :: Integer), same "ab", same (2.5 :: Double))
  let bang a = show a ++ "!"
  putStrLn (bang (2 ^ 70 :: Integer) ++ bang "x" ++ bang (1.5 :: Double) ++ bang True ++ bang (Just True))
  putStrLn (label (5 :: Integer) ++ label "x" ++ label True ++ label (3 :: Int) ++ label (Just True))
  print (norm 3 4, norm (2 ^ 40) (1 :: Integer))
  where
    norm a b = a * a + b * b
