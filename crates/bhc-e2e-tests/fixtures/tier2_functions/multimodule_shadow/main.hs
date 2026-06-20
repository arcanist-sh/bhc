module Main where

import Helper (helperResult)

go :: Int -> Int
go x = x * 2

main :: IO ()
main = do
  print helperResult
  print (go 5)
