-- Test: strictness-numeric-profile
-- Category: semantic
-- Profile: numeric
-- Expected: success
-- Spec: H26-SPEC Section 5.1

{-# PROFILE Numeric #-}

module StrictnessTest where

-- In Numeric profile, let bindings are strict by default
-- This should NOT create a thunk

test1 :: Int
test1 =
  let x = 1 + 2  -- Evaluated immediately
      y = x * 3  -- Evaluated immediately
  in y + 1       -- Result: 10

-- Lazy escape hatch
test2 :: Int
test2 =
  let x = lazy { 1 + 2 }  -- This creates a thunk
  in x + 1

-- Function application is strict
strictApp :: Int -> Int -> Int
strictApp x y = x + y  -- Both args evaluated before body

-- Pattern matching is strict in Numeric profile
testPattern :: Maybe Int -> Int
testPattern (Just x) = x  -- Scrutinee evaluated to WHNF
testPattern Nothing = 0
