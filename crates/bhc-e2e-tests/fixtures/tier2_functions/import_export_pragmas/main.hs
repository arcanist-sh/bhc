-- Exercises: a `{-# SOURCE #-}` import annotation, and an export list that
-- re-exports a qualified name (both used to be parse errors).
module Main
    ( main
    , C.toUpper
    ) where

import {-# SOURCE #-} Data.Char (ord)
import qualified Data.Char as C

main :: IO ()
main = do
    print (ord 'A')
    putStrLn [C.toUpper 'a', C.toUpper 'b']
