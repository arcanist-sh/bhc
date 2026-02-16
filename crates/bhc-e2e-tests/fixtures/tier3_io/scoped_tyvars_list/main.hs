{-# LANGUAGE ScopedTypeVariables #-}
module Main where

myReverse :: forall a. [a] -> [a]
myReverse xs = reverse (xs :: [a])

myLength :: forall a. [a] -> Int
myLength xs = length (xs :: [a])

main :: IO ()
main = do
    putStrLn (show (myReverse [1, 2, 3 :: Int]))
    putStrLn (show (myLength [10, 20, 30 :: Int]))
