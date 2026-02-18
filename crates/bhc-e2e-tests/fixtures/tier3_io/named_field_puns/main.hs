module Main where

{-# LANGUAGE NamedFieldPuns #-}

data Point = Point { x :: Int, y :: Int }

showPoint :: Point -> String
showPoint (Point { x, y }) = show x ++ "," ++ show y

makePoint :: Int -> Int -> Point
makePoint x y = Point { x, y }

addPoint :: Point -> Int
addPoint (Point { x, y }) = x + y

main :: IO ()
main = do
    let p = makePoint 3 4
    putStrLn (showPoint p)
    putStrLn (show (addPoint p))
