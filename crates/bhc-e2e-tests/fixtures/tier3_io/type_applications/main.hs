{-# LANGUAGE TypeApplications #-}
main :: IO ()
main = do
    -- Basic: id @Int
    let x = id @Int 42
    putStrLn (show x)
    -- Const with two type args
    let y = const @Int @String 10 "hello"
    putStrLn (show y)
