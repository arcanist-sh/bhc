module Main where

data Color = Red | Green | Blue
  deriving (Show, Read, Eq)

main :: IO ()
main = do
  putStrLn (show Red)
  putStrLn (show Green)
  putStrLn (show Blue)
  putStrLn (show (read "Red" :: Color))
  putStrLn (show (read "Green" :: Color))
  putStrLn (show (read "Blue" :: Color))
