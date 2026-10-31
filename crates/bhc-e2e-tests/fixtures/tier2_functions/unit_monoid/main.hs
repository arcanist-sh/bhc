-- `()` is a Semigroup and a Monoid. A writer of `()` (pandoc's
-- `AnnotatedTable` runs `RWS ([ColSpec], Int) () RowNumber`) combines its
-- output with `<>` from `mempty`; with no instance both were a stub and the
-- RWS bind crashed.
module Main where

twice :: Monoid w => w -> w
twice w = w <> w <> mempty

unitToInt :: () -> Int
unitToInt () = 7

collect :: Monoid w => [w] -> w
collect = mconcat

main :: IO ()
main = do
  print (unitToInt (mempty :: ()))
  print (unitToInt (() <> ()))
  print (unitToInt (twice ()))
  print (unitToInt (collect [(), ()]))
  print (unitToInt (mappend () ()))
  print (length (twice "ab"))
