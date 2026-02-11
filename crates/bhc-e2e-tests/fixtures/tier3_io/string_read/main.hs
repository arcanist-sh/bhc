main :: IO ()
main = do
    -- fromString is identity for String
    putStrLn (fromString "hello")
    -- read parses Int from String
    let n = read "42"
    putStrLn (show (n + 8))
    -- read with negative number
    putStrLn (show (read "-7"))
    -- readMaybe returns Just on valid input
    putStrLn (show (readMaybe "123"))
    -- readMaybe returns Nothing on invalid input
    putStrLn (show (readMaybe "abc"))
