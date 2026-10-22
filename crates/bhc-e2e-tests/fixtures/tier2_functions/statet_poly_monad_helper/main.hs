-- A `Monad n =>` helper called inside a concrete StateT do-block. The
-- monomorphizer clones it at `StateT Int IO`; its `return` must still lower
-- as StateT's (the clone used to apply IO's `pure`, so `1` ran as a closure).
module Main where

import Control.Monad.State

one :: Monad n => n Int
one = return 1

pair :: Applicative f => f (Int, Int)
pair = pure (2, 3)

step :: StateT Int IO Int
step = do
  x <- one
  (a, b) <- pair
  s <- get
  put (s + a)
  return (x + s + b)

direct :: StateT Int IO Int
direct = one

main :: IO ()
main = do
  r <- evalStateT step 41
  print r
  (v, s) <- runStateT (step >> step) 0
  print (v, s)
  d <- evalStateT direct 0
  print d
