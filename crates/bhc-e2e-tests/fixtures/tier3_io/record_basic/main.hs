module Main where

data Point = Point { x :: Int, y :: Int }

main :: IO ()
main = do
  let p = Point { x = 3, y = 4 }
  putStrLn (show (x p))
  putStrLn (show (y p))
