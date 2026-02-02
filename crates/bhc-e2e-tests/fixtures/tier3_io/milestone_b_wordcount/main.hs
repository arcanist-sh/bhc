main = do
  contents <- readFile "input.txt"
  let ls = lines contents
  let ws = words contents
  putStrLn (show (length ls))
  putStrLn (show (length ws))
  putStrLn (show (length contents))
  putStr (unlines (reverse ls))
