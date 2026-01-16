-- Test: guaranteed-fusion
-- Category: semantic
-- Profile: numeric
-- Expected: success
-- Spec: H26-SPEC Section 8.1

{-# PROFILE Numeric #-}

module FusionTest where

import H26.Tensor

-- Pattern 1: map f (map g x) MUST fuse
-- No intermediate array allocation
test1 :: Tensor Float -> Tensor Float
test1 xs = map (+1) (map (*2) xs)

-- Pattern 2: zipWith f (map g a) (map h b) MUST fuse
-- Single traversal of both arrays
test2 :: Tensor Float -> Tensor Float -> Tensor Float
test2 xs ys = zipWith (+) (map (*2) xs) (map (*3) ys)

-- Pattern 3: sum (map f x) MUST fuse
-- No intermediate array, single traversal
test3 :: Tensor Float -> Float
test3 xs = sum (map (*2) xs)

-- Pattern 4: foldl' op z (map f x) MUST fuse
-- No intermediate array, single traversal
test4 :: Tensor Float -> Float
test4 xs = foldl' (+) 0 (map (*2) xs)

-- Materialization escape hatch
test5 :: Tensor Float -> Tensor Float
test5 xs =
  let ys = materialize (map (*2) xs)  -- Force intermediate
  in map (+1) ys
