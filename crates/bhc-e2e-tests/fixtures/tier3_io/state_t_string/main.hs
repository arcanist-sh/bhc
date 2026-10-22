import Control.Monad.State

main = do
    result <- evalStateT computation "hello world"
    putStrLn result

computation :: StateT String IO String
computation = do
    s <- get
    put "done"
    return s
