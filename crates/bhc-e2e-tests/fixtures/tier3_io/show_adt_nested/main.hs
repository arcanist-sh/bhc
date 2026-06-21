module Main where

data Color = Red | Green | Blue deriving Show
data Shape = Circle Int | Rect Int Int deriving Show

main :: IO ()
main = do
  print [Red, Green, Blue]
  print (Just (Circle 5))
  print (Left Red :: Either Color Int)
  print (Red, Rect 2 3)
  print [Just (Circle 1), Nothing]
