module Main where

-- Test basic Integer (arbitrary precision) support

main :: IO ()
main = do
    -- Integer literals create BigInt values
    let x = 42 :: Integer
    let y = 100 :: Integer
    putStrLn (show x)
    putStrLn (show y)
    -- Zero
    let z = 0 :: Integer
    putStrLn (show z)
