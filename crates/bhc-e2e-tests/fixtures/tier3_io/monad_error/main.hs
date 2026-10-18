import Control.Monad.Except

main = do
    result1 <- runExceptT successComp
    case result1 of
        Left e  -> putStrLn ("Error: " ++ e)
        Right a -> putStrLn ("Success: " ++ a)
    result2 <- runExceptT failComp
    case result2 of
        Left e  -> putStrLn ("Error: " ++ e)
        Right a -> putStrLn ("Success: " ++ a)

successComp :: ExceptT String IO String
successComp = catchError (return "ok") (\e -> return ("caught: " ++ e))

failComp :: ExceptT String IO String
failComp = do
    throwError "boom"
    return "unreachable"
