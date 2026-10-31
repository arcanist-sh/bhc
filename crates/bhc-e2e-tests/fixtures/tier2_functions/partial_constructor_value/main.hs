-- An under-applied constructor used as a function value — composed, mapped,
-- or passed — builds a partial application instead of calling a stub.
-- pandoc-types' `toRow = Row nullAttr . map simpleCell` ran as
-- "stub: Row not implemented" whenever a table was built.
module Main where

data Row = Row String [Int]

data Triple = Triple Int Int Int

rowLen :: Row -> Int
rowLen (Row _ xs) = length xs

sumTriple :: Triple -> Int
sumTriple (Triple a b c) = a + b + c

applyTo :: (Int -> Triple) -> Int -> Int
applyTo f x = sumTriple (f x)

main :: IO ()
main = do
  print (map (rowLen . Row "b") [[1], [2, 3]])
  print (map (rowLen . Row "c" . map (+ 1)) [[], [4, 5, 6]])
  print (applyTo (Triple 1 2) 3)
  let mk = Triple 10
  print (sumTriple (mk 20 30))
  print (map sumTriple (zipWith (Triple 1) [2, 3] [4, 5]))
