import Control.Monad.Reader

main = do
    val <- runReaderT computation 42
    putStrLn (show val)

computation :: ReaderT Int IO Int
computation = do
    n <- ask
    return n
