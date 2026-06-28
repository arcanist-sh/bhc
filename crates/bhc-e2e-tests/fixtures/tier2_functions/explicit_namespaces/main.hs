{-# LANGUAGE ExplicitNamespaces #-}
-- ExplicitNamespaces: a `type`/`data` keyword may prefix an import/export item.
module Main
    ( main
    , type Color
    ) where

data Color = Red | Green deriving (Eq, Show)

rank :: Color -> Int
rank Red   = 0
rank Green = 1

main :: IO ()
main = do
    print (rank Green)
    print (Red == Red)
