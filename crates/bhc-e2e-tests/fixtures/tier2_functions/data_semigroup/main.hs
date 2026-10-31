-- Data.Semigroup newtypes and their instances (#52).
import Data.Semigroup

main :: IO ()
main = do
  print (getMin (Min 3 <> Min (1 :: Int)), getMax (Max 3 <> Max (9 :: Int)))
  print (getFirst (First 'a' <> First 'b'), getLast (Last 'a' <> Last 'b'))
  print (getSum (Sum 1 <> Sum (2 :: Int)))
  print (Max (4 :: Int), First (5 :: Int))
  print (getMax (foldr1 (<>) (map Max [3, 8, 2 :: Int])))
