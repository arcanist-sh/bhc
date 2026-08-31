module Main where

import Alpha (tag)
import qualified Beta as B

main :: IO ()
main = do
  print (tag 5)
  print (B.tag 2 3)
