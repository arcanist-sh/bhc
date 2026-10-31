-- A constructor imported unqualified keeps its own tag and arity when a
-- `qualified` import brings a same-named constructor of another type
-- (pandoc's HTML writer: Block's `Table` and Definition's `Cell` next to
-- `qualified AnnotatedTable as Ann`'s `Table` and `Cell`). Matching AND
-- building both in one module: the qualified one used to be registered
-- under the bare name, so a table matched `Plain`, and building the other
-- called a `stub: Table`.
module Main where
import Doc
import qualified Ann

annArea :: Ann.Table -> Int
annArea (Ann.Table a b) = a * b

describe :: Block -> String
describe (Plain n) = "plain " ++ show n
describe (Para n) = "para " ++ show n
describe (Table a b) = "table " ++ show (annArea (Ann.toTable a b))

cellSum :: Cell -> Int
cellSum (Cell a b c) = a + b + c

annCell :: Ann.Cell -> Int
annCell (Ann.Cell n) = n

main :: IO ()
main = do
  mapM_ (putStrLn . describe) [Plain 1, Table 2 3, Para 4]
  print (annArea (Ann.Table 6 7))
  print (cellSum (Cell 1 2 3))
  print (annCell (Ann.Cell 9))
