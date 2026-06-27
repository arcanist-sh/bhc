-- A module-level DEPRECATED pragma after the module name used to fail with
-- "unexpected pragma, expected `where`".
module Main {-# DEPRECATED "just a test" #-}
    ( main
    ) where

main :: IO ()
main = print (6 * 7)
