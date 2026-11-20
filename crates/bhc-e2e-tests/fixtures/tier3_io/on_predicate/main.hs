-- `on` builds a predicate from a comparison and a key. An equality built
-- this way and handed to nubBy, groupBy, unionBy or intersectBy was compiled
-- as a comparison of Bools and read an Int key as a heap object. fromJust
-- had no type, so its result printed as a number.
import Data.Char (isDigit)
import Data.Function (on)
import Data.List
import Data.Maybe

data C = R | G | B deriving (Show, Eq, Ord)

main :: IO ()
main = do
  print (nubBy ((==) `on` (`mod` 3)) [1, 2, 3, 4 :: Int])
  print (groupBy ((==) `on` (`div` 10)) [1, 2, 11, 12, 21 :: Int])
  print (unionBy ((==) `on` (`mod` 3)) [1, 2 :: Int] [3, 4, 5])
  print (intersectBy ((==) `on` (`mod` 3)) [1, 2, 3 :: Int] [4, 6])
  print (groupBy ((==) `on` isDigit) "ab12c3")
  print (groupBy ((==) `on` fst) [("a", 1 :: Int), ("a", 2), ("b", 3)])
  print (nubBy ((==) `on` snd) [(1 :: Int, R), (2, G), (3, R)])
  print (sortBy (compare `on` snd) [(1 :: Int, "b"), (2, "a")])
  print (((+) `on` length) "ab" "cde")
  print (fromJust (Just 'x'), fromJust (Just "s"), fromJust (lookup 2 [(1 :: Int, R), (2, G)]))
  print (map fromJust [Just True, Just False])
