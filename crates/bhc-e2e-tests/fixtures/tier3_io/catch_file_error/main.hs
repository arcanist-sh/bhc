main = catch (readFile "nonexistent.txt" >>= putStrLn) (\e -> putStrLn "caught error")
