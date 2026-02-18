module Main where

data StrictPair = SP !Int !Int

getFirst :: StrictPair -> Int
getFirst (SP a _) = a

getSecond :: StrictPair -> Int
getSecond (SP _ b) = b

main :: IO ()
main = do
    let p = SP 10 20
    putStrLn (show (getFirst p))
    putStrLn (show (getSecond p))
    putStrLn (show (getFirst p + getSecond p))
