import Control.Monad.State

main = do
    result <- evalStateT computation "abc"
    putStrLn result

computation :: StateT String IO String
computation = do
    s <- get
    case s of
        "" -> return "empty"
        _  -> do
            put ""
            return s
