main = do
  h <- openFile "input.txt" ReadMode
  line1 <- hGetLine h
  putStrLn line1
  line2 <- hGetLine h
  putStrLn line2
  hClose h
