main = do
  contents <- readFile "input.txt"
  writeFile "output.txt" (unlines (reverse (lines contents)))
  result <- readFile "output.txt"
  putStr result
