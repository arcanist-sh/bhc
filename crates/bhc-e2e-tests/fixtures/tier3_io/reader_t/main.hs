main = do
    let val = runReaderT ask 42
    putStrLn (show val)
