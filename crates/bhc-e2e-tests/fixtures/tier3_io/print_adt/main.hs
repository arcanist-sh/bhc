module Main where

data Color = Red | Green | Blue deriving Show
data Shape = Circle Int | Rect Int Int deriving Show

main :: IO ()
main = do
  print Red
  print Blue
  print (Circle 5)
  print (Rect 3 4)
