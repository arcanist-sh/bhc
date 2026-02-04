main = do
    let val = evalStateT computation 0
    putStrLn (show val)

computation = do
    put 10
    n <- get
    return n
