-- Importing and re-exporting the arithmetic operators (*) and (-), which lex
-- as the special Star/Minus tokens (not generic operators) and so weren't
-- accepted inside `( ... )` in import/export lists.
module Main
    ( main
    , (-)
    ) where

import Prelude (Int, IO, print, (*), (-), (+))

main :: IO ()
main = print (6 * 7 - (1 + 1))
