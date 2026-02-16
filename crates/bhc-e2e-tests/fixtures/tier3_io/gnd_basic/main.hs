{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

newtype Age = Age Int deriving (Show, Eq, Ord, Num)

showAge :: Age -> String
showAge (Age n) = "Age " ++ show n

main :: IO ()
main = do
    let a = Age 25
    let b = Age 10
    let c = a + b
    putStrLn (showAge c)
    putStrLn (showAge (a - b))
    putStrLn (showAge (a * b))
    putStrLn (show (a == b))
    putStrLn (show (a > b))
