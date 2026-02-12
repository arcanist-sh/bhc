module Main where

mySubtract :: Int -> Int -> Int
mySubtract x y = x - y

main :: IO ()
main = do
  putStrLn (show (flip mySubtract 3 10))
  putStrLn (show (flip const 1 2))
  putStrLn (show (map (flip mySubtract 5) [10, 20, 30]))
