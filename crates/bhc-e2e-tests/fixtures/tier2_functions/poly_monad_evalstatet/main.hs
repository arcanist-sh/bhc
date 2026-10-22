-- `Monad m =>` functions that run a StateT computation internally. The
-- do-block inside `evalStateT` is at `StateT s m`, so its `>>=` must not be
-- taken from the `Monad m` dictionary in scope, and the function itself is
-- not a StateT computation just because its body mentions `get`.
module Main where

import Control.Monad.State

data S = S { counter :: Int, flag :: Bool }

readFlag :: Monad m => m Bool
readFlag = evalStateT (do { s <- get; return (flag s) }) (S 5 True)

bump :: Monad m => Int -> m Int
bump k = evalStateT (do { x <- get; put (x + k); y <- get; return (x + y) }) 7

one :: Monad n => n Int
one = return 1

nested :: Monad m => m Int
nested = evalStateT (do { _ <- one; s <- get; return (counter s * 2) }) (S 21 False)

main :: IO ()
main = do
  putStrLn "start"
  b <- readFlag
  print b
  n <- bump 3
  print n
  r <- nested
  print r
