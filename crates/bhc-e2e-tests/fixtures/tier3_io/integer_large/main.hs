module Main where

-- Test Integer arithmetic and arbitrary precision

factorial :: Integer -> Integer
factorial n = if n <= 1 then 1 else n * factorial (n - 1)

main :: IO ()
main = do
    -- Basic arithmetic
    let a = 100 :: Integer
    let b = 200 :: Integer
    putStrLn (show (a + b))
    putStrLn (show (b - a))
    putStrLn (show (a * b))
    -- Factorial 20 (fits in i64 but tests recursion)
    putStrLn (show (factorial 20))
    -- Factorial 50 (overflows i64, needs arbitrary precision)
    putStrLn (show (factorial 50))
