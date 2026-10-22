module Cls (Store(..)) where

class Monad m => Store m where
  fetch :: Int -> m Int
  label :: String -> m String
  base :: m Int
