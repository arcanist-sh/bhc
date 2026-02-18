module Main where

{-# LANGUAGE InstanceSigs #-}

data Color = Red | Green | Blue

instance Show Color where
    show :: Color -> String
    show Red = "Red"
    show Green = "Green"
    show Blue = "Blue"

main :: IO ()
main = do
    putStrLn (show Red)
    putStrLn (show Green)
    putStrLn (show Blue)
