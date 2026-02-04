main = do
    let val = evalStateT get 42
    putStrLn (show val)
