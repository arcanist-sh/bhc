main :: IO ()
main = do
    putStrLn (showInt 42)
    putStrLn (showInt (0 - 7))
    putStrLn (showBool True)
    putStrLn (showBool False)
    putStrLn (showChar 'x')
