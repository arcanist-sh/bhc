main = do
    result <- evalStateT computation "hello world"
    putStrLn result

computation = do
    s <- get
    put "done"
    return s
