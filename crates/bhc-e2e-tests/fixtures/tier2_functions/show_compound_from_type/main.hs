module Main where

-- Compound values whose shape is only in the TYPE (a variable or a call),
-- not in a constructor application the show path can read.

pair :: ([Int], Int)
pair = ([1, 2], 3)

mk :: Bool -> [Int] -> ([Int], Bool)
mk flag xs = (xs, ws)
  where
    ws
      | null xs && flag = True
      | otherwise       = False

lefty :: Either Int [Int]
lefty = Left 7

righty :: Either [Bool] Char
righty = Right 'c'

nested :: Maybe (Int, [Char])
nested = Just (4, "four")

main :: IO ()
main = do
  print pair
  print (mk True [3, 1, 2])
  print (mk True [])
  print lefty
  print righty
  print nested
  print [pair, pair]
