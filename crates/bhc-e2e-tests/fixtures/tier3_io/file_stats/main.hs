main = do
  contents <- readFile "input.txt"
  putStrLn (show (length (lines contents)))
  putStrLn (show (length (words contents)))
  putStrLn (show (length contents))
