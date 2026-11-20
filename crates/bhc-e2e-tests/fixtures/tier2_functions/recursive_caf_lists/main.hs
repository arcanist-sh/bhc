-- Top-level lists defined in terms of themselves: the list is built once and
-- each element is computed once, as in GHC. `ones` segfaulted, `nats` and
-- `fibs` overflowed the stack or reported <<loop>>, and once they ran, `fibs`
-- took exponential time.
ones :: [Int]
ones = 1 : ones

nats :: [Int]
nats = 0 : map (+ 1) nats

evens, odds :: [Int]
evens = 0 : map (+ 1) odds
odds = map (+ 1) evens

fibs :: [Integer]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

primes :: [Int]
primes = sieve [2 ..]
  where
    sieve (p : xs) = p : sieve [x | x <- xs, x `mod` p /= 0]
    sieve [] = []

main :: IO ()
main = do
  print (take 3 ones)
  print (sum (take 100 ones))
  print (take 5 nats)
  print (nats !! 20000)
  print (take 4 evens, take 4 odds)
  print (take 10 fibs)
  print (fibs !! 200)
  print (length (show (fibs !! 5000)))
  print (take 10 primes)
