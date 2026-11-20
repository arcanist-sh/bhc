-- A builtin that takes a two-argument function must work when the function
-- takes one argument and returns a function. `\a -> add a` passed where
-- `\a b -> add a b` is expected had its second argument dropped: the builtin
-- got the partial application back as its result.
import qualified Data.Map as M
import Data.List
import Control.Monad

add :: Int -> Int -> Int
add a b = a + b

add3 :: Int -> Int -> Int -> Int
add3 a b c = a + b + c

cmp :: Int -> Int -> Ordering
cmp a b = compare b a

eq :: Int -> Int -> Bool
eq a b = a `mod` 3 == b `mod` 3

step :: Int -> Int -> IO Int
step a b = return (a + b)

main :: IO ()
main = do
  print (zipWith (\a -> add a) [1, 2] [10, 20])
  print (zipWith3 (\a -> add3 a) [1, 2] [10, 20] [100, 200])
  print (scanl (\a -> add a) 0 [1, 2, 3], scanr (\a -> add a) 0 [1, 2, 3], scanl1 (\a -> add a) [1, 2, 3])
  print (foldr1 (\a -> add a) [1, 2, 3], foldl1 (\a -> add a) [1, 2, 3])
  print (sortBy (\a -> cmp a) [2, 3, 1], maximumBy (\a -> cmp a) [2, 3, 1])
  print (groupBy (\a -> eq a) [1, 4, 2, 5, 3], nubBy (\a -> eq a) [1, 4, 2, 5, 3])
  print (M.toList (M.insertWith (\a -> add a) 1 5 (M.fromList [(1, 10)])))
  print (M.toList (M.unionWith (\a -> add a) (M.fromList [(1, 1)]) (M.fromList [(1, 10)])))
  print (M.toList (M.fromListWith (\a -> add a) [(1, 1), (1, 10)]))
  print (M.foldr (\a -> add a) 0 (M.fromList [(1, 1), (2, 10)]))
  print (M.foldrWithKey (\k -> add3 k) 0 (M.fromList [(1, 1), (2, 10)]))
  print (M.toList (M.mapWithKey (\k -> add k) (M.fromList [(1, 1), (2, 10)])))
  print (flip (\a -> add a) 1 2, uncurry (\a -> add a) (1, 2), map (uncurry (\a -> add a)) (zip [1, 2] [3, 4]))
  r <- foldM (\a -> step a) 0 [1, 2, 3]
  print r
  -- A function of the expected arity is passed as it is.
  print (zipWith (\a b -> add a b) [1, 2] [10, 20], zipWith add [1, 2] [10, 20], sortBy cmp [2, 3, 1])
  print (M.toList (M.insertWith (+) 1 5 (M.fromList [(1, 10)])), foldr1 add [1, 2, 3])
