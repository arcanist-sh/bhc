{-# LANGUAGE EmptyDataDecls #-}
module Main where

-- Empty data type (no constructors)
data Void

-- Data type with phantom type parameter using empty type
data Proxy a = Proxy

showProxy :: Proxy a -> String
showProxy Proxy = "Proxy"

main :: IO ()
main = do
    let p = Proxy :: Proxy Void
    putStrLn (showProxy p)
    putStrLn "done"
