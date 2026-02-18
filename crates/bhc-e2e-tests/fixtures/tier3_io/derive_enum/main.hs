{-# LANGUAGE DeriveFunctor #-}
module Main where

data Color = Red | Green | Blue deriving (Show, Eq, Enum, Bounded)

main :: IO ()
main = do
    putStrLn (show (fromEnum Red))
    putStrLn (show (fromEnum Green))
    putStrLn (show (fromEnum Blue))
    putStrLn (show (succ Red))
    putStrLn (show (pred Blue))
    putStrLn (show (toEnum 1 :: Color))
    putStrLn (show (length [Red .. Blue]))
    putStrLn (show (length [minBound .. maxBound :: Color]))
    putStrLn (show (minBound :: Color))
    putStrLn (show (maxBound :: Color))
