{-# LANGUAGE DeriveGeneric #-}
module Main where

import GHC.Generics (Generic, from, to, M1, L1, R1)

data Dir = N | S | E | W
  deriving (Show, Generic)

-- Check if a value maps to the left branch of the outermost sum
-- Use explicit nested case to avoid nested pattern compilation issues
isFirstHalf :: Dir -> Int
isFirstHalf d = case from d of
  M1 inner -> case inner of
    L1 _ -> 1
    R1 _ -> 0

main :: IO ()
main = do
  -- N and S should be in first half (L1), E and W in second half (R1)
  -- 4 constructors: split at (4+1)/2 = 2 -> L1=[N,S], R1=[E,W]
  putStrLn (show (isFirstHalf N))
  putStrLn (show (isFirstHalf S))
  putStrLn (show (isFirstHalf E))
  putStrLn (show (isFirstHalf W))
