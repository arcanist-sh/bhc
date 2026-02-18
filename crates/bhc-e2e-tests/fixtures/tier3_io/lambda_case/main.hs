module Main where

{-# LANGUAGE LambdaCase #-}

describe :: Int -> String
describe = \case
    0 -> "zero"
    1 -> "one"
    _ -> "other"

main :: IO ()
main = do
    putStrLn (describe 0)
    putStrLn (describe 1)
    putStrLn (describe 42)
