module Main where

import Control.Monad.State

-- `State s` is the mtl alias for `StateT s Identity`; runState, evalState
-- and execState all used to be unimplemented stubs.
step :: State Int Int
step = do
  x <- get
  put (x + 5)
  return (x * 2)

main :: IO ()
main = do
  let (a, s) = runState step 10
  print a
  print s
  print (evalState step 10)
  print (execState step 10)
