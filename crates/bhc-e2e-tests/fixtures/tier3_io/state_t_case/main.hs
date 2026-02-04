main = do
    result <- evalStateT computation "abc"
    putStrLn result

computation = do
    s <- get
    case s of
        "" -> return "empty"
        _  -> do
            put ""
            return s
