main :: IO ()
main = do
  putStrLn (show (compare 1 2))
  putStrLn (show (compare 5 5))
  putStrLn (show (compare 9 3))
  putStrLn (show (maximumBy compare [3, 1, 4, 1, 5, 9, 2, 6]))
  putStrLn (show (minimumBy compare [3, 1, 4, 1, 5, 9, 2, 6]))
