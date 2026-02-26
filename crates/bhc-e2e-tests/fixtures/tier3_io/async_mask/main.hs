module Main where

import Control.Exception (mask_)

main :: IO ()
main = mask_ (putStrLn "hello")
