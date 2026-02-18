-- Cross-transformer test: ExceptT over ReaderT
-- Stack: ExceptT String (ReaderT String IO) a
-- Uses: lift ask, throwE, return, runExceptT, runReaderT

comp :: ExceptT String (ReaderT String IO) Int
comp = do
    r <- lift ask
    if length r > 3
        then return (length r)
        else throwE "too short"

main :: IO ()
main = do
    result1 <- runReaderT (runExceptT comp) "Hello"
    case result1 of
        Right x -> putStrLn ("Right: " ++ show x)
        Left e  -> putStrLn ("Left: " ++ e)
    result2 <- runReaderT (runExceptT comp) "Hi"
    case result2 of
        Right x -> putStrLn ("Right: " ++ show x)
        Left e  -> putStrLn ("Left: " ++ e)
