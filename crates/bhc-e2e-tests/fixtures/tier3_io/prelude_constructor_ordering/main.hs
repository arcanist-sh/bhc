-- Bool, Ordering, Maybe and Either order by constructor and then by what the
-- constructor carries. Their values are heap objects, and the comparison
-- operators, compare, max, min, sort, maximum and minimum compared the
-- addresses: `True < False` was True and `sort [True, False, True]` came
-- back unchanged. An Ordering out of `max` or `head` printed as an address.
import Data.List (sort, sortBy, sortOn)
import Data.Ord (comparing)
data C = R | G | B deriving (Show, Eq, Ord)
main :: IO ()
main = do
  print (True < False, False < True, compare True False, max False True)
  print (Just 1 < Nothing, Nothing < Just (1 :: Int), compare (Just 3) (Just (1 :: Int)), max (Just 2) (Just (5 :: Int)))
  print (compare LT GT, GT < EQ, max LT EQ, compare () ())
  print (compare (Left 3) (Right 'a' :: Either Int Char), Left 2 < (Left 1 :: Either Int Char), Right 'a' < (Right 'b' :: Either Int Char))
  print (sort [True, False, True], sort [Just 3, Nothing, Just (1 :: Int)], sort [GT, LT, EQ])
  print (sort [Right 'b', Left 2, Right 'a', Left (1 :: Int)])
  print (minimum [True, False], maximum [False, True, False])
  print (minimum [Just 3, Nothing, Just 1], maximum [Just 3, Nothing, Just (1 :: Int)])
  print (minimum [2.5, -1.5 :: Double], maximum [LT, GT, EQ])
  print (sort [(True, 'b'), (False, 'c'), (True, 'a')], compare (Just "b", 1 :: Int) (Just "a", 2))
  print (sort [Just G, Nothing, Just R], maximum [(1 :: Int, B), (1, G)], compare [Just B] [Just R])
  print (sortOn snd [(1 :: Int, Just 'b'), (2, Nothing), (3, Just 'a')], sortBy (comparing fst) [(True, 1 :: Int), (False, 2)])
