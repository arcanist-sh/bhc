module Main where

-- Test: let-binding only evaluated when used
chooser :: Int -> Int
chooser n =
    let expensive = error "boom"
        cheap = n + 1
    in if n > 0 then cheap else expensive

main :: IO ()
main = do
    putStrLn (show (chooser 5))
    putStrLn (show (chooser 10))
