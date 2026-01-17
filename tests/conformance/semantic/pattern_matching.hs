-- Test: pattern-matching
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 3.4

{-# HASKELL_EDITION 2026 #-}

module PatternMatchingTest where

-- Simple patterns
test1 :: Int
test1 =
  let (x, y) = (1, 2)
  in x + y  -- Result: 3

-- List patterns
test2 :: Int
test2 = case [1, 2, 3] of
  []        -> 0
  [x]       -> x
  [x, y]    -> x + y
  (x:y:_)   -> x + y  -- Matches: 1 + 2 = 3

-- As-patterns
test3 :: ([Int], Int)
test3 = case [1, 2, 3] of
  xs@(x:_) -> (xs, x)  -- Returns ([1,2,3], 1)
  []       -> ([], 0)

-- View patterns (H26 feature)
test4 :: Int
test4 =
  let f (view reverse -> [3, 2, 1]) = 1
      f _                           = 0
  in f [1, 2, 3]  -- Result: 1

-- Guard patterns
test5 :: String
test5 = case 42 of
  n | n < 0     -> "negative"
    | n == 0    -> "zero"
    | n < 100   -> "small"
    | otherwise -> "large"
-- Result: "small"

-- Pattern guards with let
test6 :: Maybe Int
test6 = case Just 10 of
  Just x | let y = x * 2, y > 15 -> Just y
  _                              -> Nothing
-- Result: Just 20

-- Lazy patterns
test7 :: Int
test7 =
  let ~(x, y) = error "not evaluated"
  in 42  -- Result: 42 (pattern not forced)

-- Strict patterns (bang patterns)
test8 :: Int
test8 =
  let f !x = x + 1
  in f 41  -- Result: 42

-- Record patterns
data Point = Point { px :: Int, py :: Int }

test9 :: Int
test9 = case Point 3 4 of
  Point { px = x, py = y } -> x + y  -- Result: 7

-- Record wildcards
test10 :: Int
test10 = case Point 3 4 of
  Point {..} -> px + py  -- Result: 7 (fields bound by wildcard)

-- Nested patterns
data Tree a = Leaf a | Node (Tree a) (Tree a)

test11 :: Int
test11 = case Node (Leaf 1) (Node (Leaf 2) (Leaf 3)) of
  Leaf x                    -> x
  Node (Leaf x) (Leaf y)    -> x + y
  Node (Leaf x) (Node _ _)  -> x + 100  -- Matches: 1 + 100 = 101
  _                         -> 0

-- Literal patterns
test12 :: String
test12 = case 'a' of
  'a' -> "letter a"
  'b' -> "letter b"
  _   -> "other"
-- Result: "letter a"

-- String patterns
test13 :: Int
test13 = case "hello" of
  "hello" -> 1
  "world" -> 2
  _       -> 0
-- Result: 1

-- Numeric patterns
test14 :: String
test14 = case 3.14 of
  0.0 -> "zero"
  1.0 -> "one"
  _   -> "other"
-- Result: "other"
