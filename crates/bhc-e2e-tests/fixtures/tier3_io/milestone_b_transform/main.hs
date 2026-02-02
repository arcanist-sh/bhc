main = do
  contents <- readFile "input.txt"
  let ls = lines contents
  putStrLn (show (length ls))
  writeFile "output.txt" (unlines (reverse ls))
  result <- readFile "output.txt"
  putStr result
