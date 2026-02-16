{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Main where

newtype Meters = Meters Int deriving (Eq, Ord, Num)
newtype Seconds = Seconds Int deriving (Eq, Ord)

showMeters :: Meters -> String
showMeters (Meters n) = show n ++ "m"

showSeconds :: Seconds -> String
showSeconds (Seconds n) = show n ++ "s"

main :: IO ()
main = do
    let d1 = Meters 100
    let d2 = Meters 50
    putStrLn (showMeters (d1 + d2))
    putStrLn (showMeters (d1 - d2))
    putStrLn (showMeters (d1 * d2))
    let t1 = Seconds 30
    let t2 = Seconds 20
    putStrLn (show (t1 == t2))
    putStrLn (show (t1 > t2))
    putStrLn (show (t1 < t2))
