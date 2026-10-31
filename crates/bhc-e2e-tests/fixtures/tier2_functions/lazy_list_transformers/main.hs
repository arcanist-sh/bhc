-- map/filter/concatMap/zip/zipWith over infinite lists stop at an
-- unevaluated tail and resume lazily (#54).
from :: Int -> [Int]
from n = n : from (n + 1)

sieve :: [Int] -> [Int]
sieve (p : xs) = p : sieve [x | x <- xs, x `mod` p /= 0]
sieve [] = []

main :: IO ()
main = do
  print (take 3 (map (* 2) (from 1)))
  print (takeWhile (< 20) (map (* 3) (map (+ 1) (from 0))))
  print (take 4 (filter even (from 1)))
  print (head (filter (> 1000) (from 0)))
  print (take 3 [x * 2 | x <- from 1, odd x])
  print (take 3 [(x, y) | x <- from 1, y <- "ab"])
  print (map fst (take 3 (zip (from 1) (repeat 'z'))))
  print (take 3 (zipWith (+) (from 1) (from 100)))
  print (take 6 (sieve (from 2)))
  print (map (+ 1) [1, 2, 3 :: Int], filter odd [1 .. 10 :: Int])
  print (sum (map (* 2) [1 .. 100000 :: Int]))
  print (length (filter even (map (+ 1) (take 100000 (from 0)))))
