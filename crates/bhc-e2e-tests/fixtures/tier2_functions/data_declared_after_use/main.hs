-- A function above its data type's declaration matches and builds that
-- type's constructors with their real tags. pandoc's HTML writer matches
-- `TablePart` in `tablePartToHtml` and declares it further down; the case
-- switched on made-up tags and a table crashed the writer.
module Main where

pick :: Part -> Int
pick p =
  let t = case p of
        Thead -> 10
        Tfoot -> 20
        Tbody -> 30
  in t + 1

describe :: Shape -> String
describe (Circle r) = "circle " ++ show r
describe (Rect w h) = "rect " ++ show (w * h)
describe Dot = "dot"

shapes :: [Shape]
shapes = [Dot, Rect 2 3, Circle 5]

data Part = Thead | Tfoot | Tbody
  deriving (Eq)

data Shape = Circle Int | Rect Int Int | Dot

main :: IO ()
main = do
  print (pick Thead)
  print (pick Tfoot)
  print (pick Tbody)
  print (Tfoot == Tfoot)
  mapM_ (putStrLn . describe) shapes
