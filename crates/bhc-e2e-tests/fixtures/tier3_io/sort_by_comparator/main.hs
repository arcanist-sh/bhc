-- `compare` answers with an `Ordering`, boxed or as a tagged immediate.
-- Reading the ADDRESS of a boxed one as a signed ordering made every
-- comparison "greater", so `sortBy compare` left its list as it found it.
module Main where
import Data.List (sortBy, sortOn, maximumBy, minimumBy)
import Data.Ord (comparing)

main :: IO ()
main = do
  print (sortBy compare [3, 1, 2 :: Int])
  print (sortBy (\a b -> compare b a) [3, 1, 2 :: Int])
  print (map fst (sortBy (comparing snd) [(1 :: Int, 'b'), (2 :: Int, 'a')]))
  print (sortOn negate [3, 1, 2 :: Int])
  print (maximumBy compare [3, 1, 2 :: Int])
  print (minimumBy compare [3, 1, 2 :: Int])
