-- A module that hides Prelude names and defines its own, still reaching the
-- Prelude through a qualified name (#158).
module Main where

import Prelude hiding (show, (==), max, lookup)
import qualified Prelude
import qualified Prelude as P

show :: (Int -> Int) -> String
show f = "f 1 = " ++ Prelude.show (f 1)

(==) :: (Int -> Int) -> (Int -> Int) -> Bool
f == g = f 0 Prelude.== g 0

max :: Int -> Int -> Int
max a b = a - b

lookup :: Int -> String
lookup n = replicate n 'x' ++ P.show (P.max n 10)

main :: IO ()
main = do
  putStrLn (show (+ 1))
  print ((+ 1) == (+ 1), (+ 1) == (+ 2))
  print ((==) (subtract 1) (+ (-1)))
  print (max 10 3, 10 `max` 3, P.max 10 3)
  putStrLn (lookup 3)
  print (P.lookup 2 [(1, "a"), (2, "b")])
  print (Prelude.show (3 :: Int), 3 Prelude.== (3 :: Int), "a" P.== "b")
  print (map Prelude.show [1, 2 :: Int])
