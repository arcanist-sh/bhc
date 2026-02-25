module Main where

main :: IO ()
main = do
    let stock = 10
    let family = "Smith"
    let via = True
    let role = 99
    let strict = stock + role
    let lazy = 7
    let pattern = 3
    let anyclass = 5
    putStrLn (show strict)
    putStrLn family
    putStrLn (show via)
    putStrLn (show lazy)
    putStrLn (show pattern)
    putStrLn (show anyclass)
