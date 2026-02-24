{-# LANGUAGE DeriveGeneric #-}
module Main where

import GHC.Generics (Generic, from, to)

data Color = Red | Green | Blue
  deriving (Show, Generic)

data Pair a b = MkPair a b
  deriving (Show, Generic)

main :: IO ()
main = do
  -- Roundtrip enum types
  putStrLn (show (to (from Red)))
  putStrLn (show (to (from Green)))
  putStrLn (show (to (from Blue)))
  -- Roundtrip product type
  putStrLn (show (to (from (MkPair 1 2))))
