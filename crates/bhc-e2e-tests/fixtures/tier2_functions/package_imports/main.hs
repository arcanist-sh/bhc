{-# LANGUAGE PackageImports #-}
-- PackageImports: a package-name string between `import [qualified]` and the
-- module name. BHC resolves by module name, so the package string is skipped.
module Main where

import "base" Data.Char (ord)
import qualified "base" Data.Char as C

main :: IO ()
main = do
    print (ord 'A')
    putStrLn [C.chr 66, C.chr 67]
