-- Test: recursion
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 3.5

{-# HASKELL_EDITION 2026 #-}

module RecursionTest where

-- ================================================================
-- Simple Recursion (Top-level)
-- ================================================================

-- Factorial
factorial :: Int -> Int
factorial n = case n of
  0 -> 1
  _ -> n * factorial (n - 1)

testFactorial0 :: Int
testFactorial0 = factorial 0
-- Result: 1

testFactorial1 :: Int
testFactorial1 = factorial 1
-- Result: 1

testFactorial5 :: Int
testFactorial5 = factorial 5
-- Result: 120

testFactorial10 :: Int
testFactorial10 = factorial 10
-- Result: 3628800

-- Fibonacci
fib :: Int -> Int
fib n = case n of
  0 -> 0
  1 -> 1
  _ -> fib (n - 1) + fib (n - 2)

testFib0 :: Int
testFib0 = fib 0
-- Result: 0

testFib1 :: Int
testFib1 = fib 1
-- Result: 1

testFib5 :: Int
testFib5 = fib 5
-- Result: 5

testFib10 :: Int
testFib10 = fib 10
-- Result: 55

-- ================================================================
-- List Recursion
-- ================================================================

-- Sum of a list
sumList :: [Int] -> Int
sumList xs = case xs of
  []     -> 0
  (y:ys) -> y + sumList ys

testSum0 :: Int
testSum0 = sumList []
-- Result: 0

testSum5 :: Int
testSum5 = sumList [1, 2, 3, 4, 5]
-- Result: 15

-- Product of a list
productList :: [Int] -> Int
productList xs = case xs of
  []     -> 1
  (y:ys) -> y * productList ys

testProduct0 :: Int
testProduct0 = productList []
-- Result: 1

testProduct5 :: Int
testProduct5 = productList [1, 2, 3, 4, 5]
-- Result: 120

-- Length of a list
lengthList :: [a] -> Int
lengthList xs = case xs of
  []     -> 0
  (_:ys) -> 1 + lengthList ys

testLength0 :: Int
testLength0 = lengthList ([] :: [Int])
-- Result: 0

testLength5 :: Int
testLength5 = lengthList [1, 2, 3, 4, 5]
-- Result: 5

-- Map over a list
mapList :: (a -> b) -> [a] -> [b]
mapList f xs = case xs of
  []     -> []
  (y:ys) -> f y : mapList f ys

testMapLength :: Int
testMapLength = lengthList (mapList (\x -> x + 1) [1, 2, 3])
-- Result: 3

-- Filter a list
filterList :: (a -> Bool) -> [a] -> [a]
filterList p xs = case xs of
  []     -> []
  (y:ys) -> case p y of
    True  -> y : filterList p ys
    False -> filterList p ys

testFilterLength :: Int
testFilterLength = lengthList (filterList (\x -> x > 2) [1, 2, 3, 4, 5])
-- Result: 3

-- Append two lists
appendList :: [a] -> [a] -> [a]
appendList xs ys = case xs of
  []     -> ys
  (z:zs) -> z : appendList zs ys

testAppend :: Int
testAppend = lengthList (appendList [1, 2, 3] [4, 5])
-- Result: 5

-- Reverse a list
reverseList :: [a] -> [a]
reverseList xs = reverseAcc xs []
  where
    reverseAcc as acc = case as of
      []     -> acc
      (y:ys) -> reverseAcc ys (y : acc)

testReverse :: Int
testReverse = sumList (reverseList [1, 2, 3])
-- Result: 6 (same sum, different order)

-- ================================================================
-- Tail Recursion
-- ================================================================

-- Tail-recursive factorial
factorialTail :: Int -> Int
factorialTail n = factorialAcc n 1
  where
    factorialAcc m acc = case m of
      0 -> acc
      _ -> factorialAcc (m - 1) (acc * m)

testFactorialTail5 :: Int
testFactorialTail5 = factorialTail 5
-- Result: 120

-- Tail-recursive sum
sumTail :: [Int] -> Int
sumTail xs = sumAcc xs 0
  where
    sumAcc ys acc = case ys of
      []     -> acc
      (z:zs) -> sumAcc zs (acc + z)

testSumTail5 :: Int
testSumTail5 = sumTail [1, 2, 3, 4, 5]
-- Result: 15

-- ================================================================
-- Local Recursive Functions (let rec)
-- ================================================================

-- Local factorial
testLocalFactorial :: Int
testLocalFactorial =
  let fact n = case n of
        0 -> 1
        _ -> n * fact (n - 1)
  in fact 5
-- Result: 120

-- Local fib
testLocalFib :: Int
testLocalFib =
  let fib n = case n of
        0 -> 0
        1 -> 1
        _ -> fib (n - 1) + fib (n - 2)
  in fib 10
-- Result: 55

-- Local countdown
testLocalCountdown :: Int
testLocalCountdown =
  let countdown n acc = case n of
        0 -> acc
        _ -> countdown (n - 1) (acc + n)
  in countdown 10 0
-- Result: 55 (sum of 1..10)

-- ================================================================
-- Mutual Recursion
-- ================================================================

-- Even/odd check
isEven :: Int -> Bool
isEven n = case n of
  0 -> True
  _ -> isOdd (n - 1)

isOdd :: Int -> Bool
isOdd n = case n of
  0 -> False
  _ -> isEven (n - 1)

testIsEven0 :: Bool
testIsEven0 = isEven 0
-- Result: True

testIsEven4 :: Bool
testIsEven4 = isEven 4
-- Result: True

testIsOdd3 :: Bool
testIsOdd3 = isOdd 3
-- Result: True

testIsOdd4 :: Bool
testIsOdd4 = isOdd 4
-- Result: False

-- ================================================================
-- GCD (Euclidean algorithm)
-- ================================================================

gcd' :: Int -> Int -> Int
gcd' a b = case b of
  0 -> a
  _ -> gcd' b (a `mod` b)

testGcd1 :: Int
testGcd1 = gcd' 12 8
-- Result: 4

testGcd2 :: Int
testGcd2 = gcd' 48 18
-- Result: 6

testGcd3 :: Int
testGcd3 = gcd' 17 5
-- Result: 1 (coprime)

-- ================================================================
-- Power function
-- ================================================================

power :: Int -> Int -> Int
power base exp = case exp of
  0 -> 1
  _ -> base * power base (exp - 1)

testPower1 :: Int
testPower1 = power 2 0
-- Result: 1

testPower2 :: Int
testPower2 = power 2 10
-- Result: 1024

testPower3 :: Int
testPower3 = power 3 4
-- Result: 81

-- ================================================================
-- Main function to run all tests
-- ================================================================

main :: IO ()
main = do
  -- Factorial
  print testFactorial0    -- Expected: 1
  print testFactorial5    -- Expected: 120
  print testFactorial10   -- Expected: 3628800

  -- Fibonacci
  print testFib0          -- Expected: 0
  print testFib5          -- Expected: 5
  print testFib10         -- Expected: 55

  -- List operations
  print testSum5          -- Expected: 15
  print testProduct5      -- Expected: 120
  print testLength5       -- Expected: 5
  print testAppend        -- Expected: 5

  -- Tail recursion
  print testFactorialTail5  -- Expected: 120
  print testSumTail5        -- Expected: 15

  -- Local recursion
  print testLocalFactorial  -- Expected: 120
  print testLocalFib        -- Expected: 55

  -- Mutual recursion
  print testIsEven4       -- Expected: True (1)
  print testIsOdd3        -- Expected: True (1)

  -- GCD
  print testGcd1          -- Expected: 4
  print testGcd2          -- Expected: 6

  -- Power
  print testPower2        -- Expected: 1024
  print testPower3        -- Expected: 81
