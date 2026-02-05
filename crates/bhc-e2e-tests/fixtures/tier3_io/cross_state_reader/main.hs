-- Cross-transformer test: StateT with ReaderT operations
-- Tests MonadReader r m => MonadReader r (StateT s m) lifted instance
--
-- Stack: StateT Int (ReaderT String IO) a
-- Uses: get (direct), ask (lifted from ReaderT)

main :: IO ()
main = do
    result <- runReaderT (evalStateT comp 10) "Hello"
    putStrLn (show result)

-- Uses both StateT and ReaderT operations
comp :: StateT Int (ReaderT String IO) Int
comp = do
    s <- get       -- MonadState operation (direct)
    r <- ask       -- MonadReader operation (lifted)
    return (s + length r)
