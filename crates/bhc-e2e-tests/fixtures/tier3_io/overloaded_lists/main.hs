{-# LANGUAGE OverloadedLists #-}
main :: IO ()
main = do
  let xs = [1, 2, 3] :: [Int]
  putStrLn (show (length xs))
  putStrLn (show (sum xs))
