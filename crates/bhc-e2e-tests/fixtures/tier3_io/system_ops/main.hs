main = do
  r1 <- doesFileExist "input.txt"
  putStrLn (if r1 then "True" else "False")
  r2 <- doesDirectoryExist "input.txt"
  putStrLn (if r2 then "True" else "False")
