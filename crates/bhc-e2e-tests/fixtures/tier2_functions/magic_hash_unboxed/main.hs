{-# LANGUAGE MagicHash #-}
-- Boxed approximation of MagicHash unboxed semantics: I# is an identity
-- wrapper, the primops (+#/-#/*#/<#) map to their boxed Int operations, and
-- `1#`/`2#` literals are ordinary Int literals. The program should typecheck,
-- lower, and run identically on both backends — no real unboxing, just
-- semantics-preserving desugaring in bhc-lower.
module Main where

import GHC.Exts

bump :: Int -> Int
bump (I# n) = I# (n +# 1#)

double :: Int -> Int
double (I# n) = I# (n *# 2#)

diff :: Int -> Int -> Int
diff (I# a) (I# b) = I# (a -# b)

lessThan :: Int -> Int -> Bool
lessThan (I# a) (I# b) = a <# b

main :: IO ()
main = do
    print (bump 41)
    print (double 21)
    print (diff 50 8)
    print (lessThan 3 5)
    print (5 +# 3#)
