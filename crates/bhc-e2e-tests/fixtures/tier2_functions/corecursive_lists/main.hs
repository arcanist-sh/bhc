-- Corecursive producers and the infinite Prelude producers, consumed by
-- finite consumers (#28).
import Prelude hiding (repeat)

nats :: Int -> [Int]
nats n = n : nats (n + 1)

evens :: [Int]
evens = go 0
  where go k = k : go (k + 2)

letters :: Char -> String
letters c = c : letters (succ c)

collatz :: Int -> [Int]
collatz 1 = [1]
collatz n = n : collatz (if even n then n `div` 2 else 3 * n + 1)

-- A local definition shadows the builtin of the same name.
repeat :: Int -> [Int]
repeat n = [n, n]

main :: IO ()
main = do
  print (take 3 (nats 5))
  print (head (tail (nats 5)))
  print (nats 0 !! 10)
  print (takeWhile (< 4) (nats 0))
  print (take 4 evens)
  print (zip [1, 2, 3 :: Int] (nats 10))
  print (take 3 (drop 5 (nats 0)))
  putStrLn (take 5 (letters 'a'))
  print (length (collatz 27))
  case nats 7 of
    (_ : ys) -> print (take 2 ys)
    [] -> pure ()
  print (iterate (* 2) (1 :: Int) !! 10)
  print (takeWhile (< 100) (iterate (* 3) (1 :: Int)))
  print (zip [0 :: Int ..] "abc")
  print (take 7 (cycle [1, 2, 3 :: Int]))
  print (repeat 3)
  print (sum (take 5 (map (* 2) (take 10 (nats 1)))))
