{-# LANGUAGE FlexibleContexts #-}
module Main where

import Control.Monad.Reader

-- `Reader r` is the mtl alias for `ReaderT r Identity`. Its runner used to
-- be an unimplemented stub, so any program using the non-transformer
-- spelling trapped at runtime.
simple :: Reader Int Int
simple = do
  x <- ask
  return (x + 1)

viaAsksAndLocal :: Reader Int Int
viaAsksAndLocal = do
  a <- asks (+ 1)
  b <- local (* 2) ask
  return (a + b)

main :: IO ()
main = do
  print (runReader simple 41)
  print (runReader viaAsksAndLocal 10)
