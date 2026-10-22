-- A lambda whose first parameter is a pattern takes all its arguments at
-- once: a builtin fold applies `\(a, b) x -> ...` to both together.
import Data.List (foldl')

sumAndCount :: [Int] -> (Int, Int)
sumAndCount = foldl' (\(s, n) x -> (s + x, n + 1)) (0, 0)

main :: IO ()
main = do
  print (fst (foldl (\(a, b) x -> (a + x, b)) (0 :: Int, 0 :: Int) [1, 2, 3]))
  let (s, n) = sumAndCount [4, 5, 6]
  print s
  print n
  let (lo, hi) = foldr (\x (l, h) -> (min x l, max x h)) (100, 0) [7, 3, 9 :: Int]
  print lo
  print hi
