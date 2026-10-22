import Control.Monad.State

main = do
    val <- evalStateT computation 0
    putStrLn (show val)

computation :: StateT Int IO Int
computation = do
    put 10
    n <- get
    return n
