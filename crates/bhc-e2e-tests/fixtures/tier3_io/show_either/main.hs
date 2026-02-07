main :: IO ()
main = do
    putStrLn (show (Left 42 :: Either Int Bool))
    putStrLn (show (Right True :: Either Int Bool))
