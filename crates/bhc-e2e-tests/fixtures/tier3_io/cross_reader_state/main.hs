-- Cross-transformer test: ReaderT with StateT operations
-- Tests MonadState s m => MonadState s (ReaderT r m) lifted instance
--
-- Stack: ReaderT String (StateT Int IO) a
-- Uses: ask (direct), get/put (lifted from StateT)

main :: IO ()
main = do
    (result, finalState) <- runStateT (runReaderT comp "World") 5
    putStrLn (show result)
    putStrLn (show finalState)

-- Uses both ReaderT and StateT operations
comp :: ReaderT String (StateT Int IO) Int
comp = do
    r <- ask       -- MonadReader operation (direct)
    s <- get       -- MonadState operation (lifted)
    put (s + 1)    -- MonadState operation (lifted)
    return (length r + s)
