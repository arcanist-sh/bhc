main :: IO ()
main = do
    putStrLn (show (Just 42))
    putStrLn (show (Nothing :: Maybe Int))
    putStrLn (show (Just True))
