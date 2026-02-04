main = do
    let val = runReaderT computation 42
    putStrLn (show val)

computation = do
    n <- ask
    return n
