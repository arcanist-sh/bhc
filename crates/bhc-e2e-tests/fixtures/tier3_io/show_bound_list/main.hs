-- `show` on a list held in a variable. A list literal rendered correctly, but
-- a bound one printed the LIST's address, because lists were left out of the
-- type annotation that ADTs got, and the element descriptor defaulted to Int
-- when there was no cons cell to read.
module Main where

data C = C Int | D String deriving Show

main :: IO ()
main = do
  print [1, 2 :: Int]
  print "hi"
  print ['a', 'b']
  print (Just [1, 2 :: Int])
  print [Just (1 :: Int), Nothing]
  print (Right 5 :: Either String Int)
  print [(1 :: Int, 'x')]
  let strs = ["ab", "cd"]
  print strs
  let ints = [1, 2 :: Int]
  print ints
  let cons = [C 1, D "s"]
  print cons
