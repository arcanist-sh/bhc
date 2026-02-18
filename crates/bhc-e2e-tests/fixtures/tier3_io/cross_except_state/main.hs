-- Cross-transformer test: ExceptT over StateT
-- Stack: ExceptT String (StateT Int IO) a
-- Uses: lift get, lift (put ...), throwE, return, runExceptT, runStateT

comp :: ExceptT String (StateT Int IO) Int
comp = do
    s <- lift get
    lift (put (s + 1))
    if s > 3
        then throwE "too big"
        else return (s * 10)

main :: IO ()
main = do
    (result1, state1) <- runStateT (runExceptT comp) 2
    case result1 of
        Right x -> putStrLn ("Right: " ++ show x)
        Left e  -> putStrLn ("Left: " ++ e)
    putStrLn (show state1)
    (result2, state2) <- runStateT (runExceptT comp) 5
    case result2 of
        Right x -> putStrLn ("Right: " ++ show x)
        Left e  -> putStrLn ("Left: " ++ e)
    putStrLn (show state2)
