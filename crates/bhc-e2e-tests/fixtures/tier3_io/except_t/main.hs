import Control.Monad.Trans.Except

main = do
    result1 <- runExceptT successComp
    case result1 of
        Left e  -> putStrLn ("Left: " ++ e)
        Right a -> putStrLn ("Right: " ++ a)
    result2 <- runExceptT failComp
    case result2 of
        Left e  -> putStrLn ("Left: " ++ e)
        Right a -> putStrLn ("Right: " ++ a)

successComp :: ExceptT String IO String
successComp = catchE (return "success") (\_ -> return "caught")

failComp :: ExceptT String IO String
failComp = do
    throwE "failed"
    return "unreachable"
