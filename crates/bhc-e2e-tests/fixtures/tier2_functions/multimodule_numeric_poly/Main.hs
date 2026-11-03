-- Class-polymorphic functions from another module used at Integer and
-- Double: each use gets a copy of the imported body at its type (#57).
module Main where

import Sq

main :: IO ()
main = do
  print (sq (3 :: Integer), sq (2 ^ 40 :: Integer), sq (1.5 :: Double), sq (4 :: Int))
  print (norm (2 ^ 40) (1 :: Integer), norm 1.5 (2 :: Double))
  print (bigger (2 ^ 70) (5 :: Integer), bigger 2.5 (1 :: Double))
