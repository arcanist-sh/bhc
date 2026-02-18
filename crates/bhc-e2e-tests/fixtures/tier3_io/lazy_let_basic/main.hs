module Main where

-- Test: unused let-binding with error should not crash
main :: IO ()
main = do
    let x = error "should not be evaluated"
    let y = 42
    putStrLn (show y)
