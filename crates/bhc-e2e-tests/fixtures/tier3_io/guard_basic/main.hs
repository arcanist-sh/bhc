main :: IO ()
main = do
    putStrLn (show (length (guard True)))
    putStrLn (show (length (guard False)))
