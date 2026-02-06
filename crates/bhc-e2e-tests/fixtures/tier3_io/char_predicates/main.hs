main :: IO ()
main = do
    putStrLn (showBool (isAlpha 'A'))
    putStrLn (showBool (isAlpha '1'))
    putStrLn (showBool (isDigit '5'))
    putStrLn (showBool (isDigit 'x'))
    putStrLn (showBool (isSpace ' '))
    putStrLn (showBool (isUpper 'A'))
    putStrLn (showBool (isLower 'a'))
    let u = toUpper 'a'
    let l = toLower 'Z'
    putStrLn (showChar u)
    putStrLn (showChar l)
