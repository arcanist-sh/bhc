{-# LANGUAGE MagicHash #-}
-- MagicHash lets identifiers/constructors end in `#`. Here they're ordinary
-- (no unboxed semantics) — the point is that the names lex and parse.
module Main where

data Box = Box# Int deriving (Eq, Show)

unbox# :: Box -> Int
unbox# (Box# n) = n

addBoxes# :: Box -> Box -> Box
addBoxes# (Box# a) (Box# b) = Box# (a + b)

main :: IO ()
main = do
    print (unbox# (Box# 42))
    print (unbox# (addBoxes# (Box# 1) (Box# 2)))
    print (Box# 5 == Box# 5)
