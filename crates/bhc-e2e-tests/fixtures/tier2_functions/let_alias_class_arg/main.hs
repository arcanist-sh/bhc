-- A let-bound alias of a function argument keeps the argument's type, so a
-- class-constrained function mapped over it gets its dictionary. pandoc's
-- pipe-table reader (`let heads' = take numcols heads` then `map realLength
-- heads'`) ran realLength with none: once because the let was generalized
-- over a variable the environment still held under its old name, and once
-- because unification compared against a half-resolved type and overwrote
-- the alias's element type.
module Main where

data Cell = Cell String

class Width a where
  width :: a -> Int

instance Width Cell where
  width (Cell s) = length s

totalWidth :: Width a => a -> Int
totalWidth x = width x + 1

viaTake :: [Cell] -> [Int] -> [Int]
viaTake cells cols =
  let n = length cols
      cells' = take n cells
  in map totalWidth cells'

viaAlias :: [Cell] -> [Int]
viaAlias cells = let cells' = cells in map totalWidth cells'

viaDo :: Monad m => m [Cell] -> m [Int]
viaDo mcells = do
  cells <- mcells
  let cells' = cells
  return (map totalWidth cells')

main :: IO ()
main = do
  let cs = [Cell "ab", Cell "cde", Cell ""]
  print (viaTake cs [0, 0])
  print (viaAlias cs)
  r <- viaDo (return cs)
  print r
