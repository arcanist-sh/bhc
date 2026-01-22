-- Test: string-io
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 3.8

{-# HASKELL_EDITION 2026 #-}

module StringIOTest where

-- ================================================================
-- Basic putStrLn (print string with newline)
-- ================================================================

testPutStrLn1 :: IO ()
testPutStrLn1 = putStrLn "Hello, World!"
-- Output: Hello, World!

testPutStrLn2 :: IO ()
testPutStrLn2 = putStrLn "BHC Compiler"
-- Output: BHC Compiler

testPutStrLnEmpty :: IO ()
testPutStrLnEmpty = putStrLn ""
-- Output: (empty line)

-- ================================================================
-- Basic putStr (print string without newline)
-- ================================================================

testPutStr1 :: IO ()
testPutStr1 = putStr "Hello"
-- Output: Hello (no newline)

testPutStr2 :: IO ()
testPutStr2 = do
  putStr "Hello, "
  putStrLn "World!"
-- Output: Hello, World!

-- ================================================================
-- putChar (print single character)
-- ================================================================

testPutChar1 :: IO ()
testPutChar1 = putChar 'A'
-- Output: A

testPutChar2 :: IO ()
testPutChar2 = do
  putChar 'H'
  putChar 'i'
  putChar '\n'
-- Output: Hi

testPutCharNewline :: IO ()
testPutCharNewline = putChar '\n'
-- Output: (newline)

-- ================================================================
-- print (show and print values)
-- ================================================================

-- Print integers
testPrintInt1 :: IO ()
testPrintInt1 = print 42
-- Output: 42

testPrintInt2 :: IO ()
testPrintInt2 = print (-123)
-- Output: -123

testPrintInt3 :: IO ()
testPrintInt3 = print 0
-- Output: 0

-- Print large integers
testPrintIntLarge :: IO ()
testPrintIntLarge = print 1000000
-- Output: 1000000

-- Print booleans
testPrintBoolTrue :: IO ()
testPrintBoolTrue = print True
-- Output: True

testPrintBoolFalse :: IO ()
testPrintBoolFalse = print False
-- Output: False

-- ================================================================
-- Sequential IO operations
-- ================================================================

testSequence1 :: IO ()
testSequence1 = do
  putStrLn "Line 1"
  putStrLn "Line 2"
  putStrLn "Line 3"
-- Output:
-- Line 1
-- Line 2
-- Line 3

testSequence2 :: IO ()
testSequence2 = do
  print 1
  print 2
  print 3
-- Output:
-- 1
-- 2
-- 3

testMixed :: IO ()
testMixed = do
  putStr "The answer is: "
  print 42
-- Output: The answer is: 42

-- ================================================================
-- IO with pure computation
-- ================================================================

double :: Int -> Int
double x = x * 2

testPureComputation :: IO ()
testPureComputation = do
  let x = double 21
  print x
-- Output: 42

factorial :: Int -> Int
factorial n = case n of
  0 -> 1
  _ -> n * factorial (n - 1)

testFactorialPrint :: IO ()
testFactorialPrint = do
  print (factorial 5)
-- Output: 120

-- ================================================================
-- IO with conditionals
-- ================================================================

printSign :: Int -> IO ()
printSign n =
  if n < 0
    then putStrLn "negative"
    else if n > 0
      then putStrLn "positive"
      else putStrLn "zero"

testSign1 :: IO ()
testSign1 = printSign (-5)
-- Output: negative

testSign2 :: IO ()
testSign2 = printSign 10
-- Output: positive

testSign3 :: IO ()
testSign3 = printSign 0
-- Output: zero

-- ================================================================
-- IO with case expressions
-- ================================================================

describeMaybe :: Maybe Int -> IO ()
describeMaybe mx = case mx of
  Nothing -> putStrLn "Nothing here"
  Just x  -> do
    putStr "Got value: "
    print x

testDescribeJust :: IO ()
testDescribeJust = describeMaybe (Just 42)
-- Output: Got value: 42

testDescribeNothing :: IO ()
testDescribeNothing = describeMaybe Nothing
-- Output: Nothing here

-- ================================================================
-- Printing results of list operations
-- ================================================================

testPrintLength :: IO ()
testPrintLength = print (length [1, 2, 3, 4, 5])
-- Output: 5

testPrintHead :: IO ()
testPrintHead = print (head [10, 20, 30])
-- Output: 10

testPrintSum :: IO ()
testPrintSum = do
  let xs = [1, 2, 3, 4, 5]
  let total = sumList xs
  putStr "Sum: "
  print total
  where
    sumList :: [Int] -> Int
    sumList ys = case ys of
      []     -> 0
      (z:zs) -> z + sumList zs
-- Output: Sum: 15

-- ================================================================
-- Multiple values on same line
-- ================================================================

printTwo :: Int -> Int -> IO ()
printTwo x y = do
  print x
  print y

testPrintTwo :: IO ()
testPrintTwo = printTwo 1 2
-- Output:
-- 1
-- 2

printThree :: Int -> Int -> Int -> IO ()
printThree x y z = do
  print x
  print y
  print z

testPrintThree :: IO ()
testPrintThree = printThree 10 20 30
-- Output:
-- 10
-- 20
-- 30

-- ================================================================
-- Looping with IO
-- ================================================================

printRange :: Int -> Int -> IO ()
printRange lo hi =
  if lo > hi
    then pure ()
    else do
      print lo
      printRange (lo + 1) hi

testPrintRange :: IO ()
testPrintRange = printRange 1 5
-- Output:
-- 1
-- 2
-- 3
-- 4
-- 5

-- Print list elements
printList :: [Int] -> IO ()
printList xs = case xs of
  []     -> pure ()
  (y:ys) -> do
    print y
    printList ys

testPrintList :: IO ()
testPrintList = printList [100, 200, 300]
-- Output:
-- 100
-- 200
-- 300

-- ================================================================
-- String building and printing
-- ================================================================

greet :: String -> IO ()
greet name = do
  putStr "Hello, "
  putStr name
  putStrLn "!"

testGreet :: IO ()
testGreet = greet "BHC"
-- Output: Hello, BHC!

-- ================================================================
-- Main function to run all tests
-- ================================================================

main :: IO ()
main = do
  -- Basic putStrLn
  putStrLn "=== putStrLn tests ==="
  testPutStrLn1
  testPutStrLn2
  testPutStrLnEmpty

  -- Basic putStr
  putStrLn "=== putStr tests ==="
  testPutStr2

  -- putChar
  putStrLn "=== putChar tests ==="
  testPutChar2

  -- print integers
  putStrLn "=== print Int tests ==="
  testPrintInt1
  testPrintInt2
  testPrintInt3
  testPrintIntLarge

  -- print booleans
  putStrLn "=== print Bool tests ==="
  testPrintBoolTrue
  testPrintBoolFalse

  -- sequences
  putStrLn "=== sequence tests ==="
  testSequence1
  testSequence2
  testMixed

  -- pure computation
  putStrLn "=== pure computation tests ==="
  testPureComputation
  testFactorialPrint

  -- conditionals
  putStrLn "=== conditional tests ==="
  testSign1
  testSign2
  testSign3

  -- case expressions
  putStrLn "=== case expression tests ==="
  testDescribeJust
  testDescribeNothing

  -- list operations
  putStrLn "=== list operation tests ==="
  testPrintLength
  testPrintHead
  testPrintSum

  -- multiple values
  putStrLn "=== multiple value tests ==="
  testPrintTwo
  testPrintThree

  -- loops
  putStrLn "=== loop tests ==="
  testPrintRange
  testPrintList

  -- string building
  putStrLn "=== string building tests ==="
  testGreet

  putStrLn "=== All tests completed ==="
