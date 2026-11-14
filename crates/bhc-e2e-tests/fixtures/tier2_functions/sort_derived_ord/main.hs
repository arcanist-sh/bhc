-- Sorting and extremes at types with a derived Ord, and the Data.List /
-- Data.Ord / Data.Function helpers around them. `sort`, `maximum` and
-- `minimum` compared boxed values by address; `import Data.List` typed `sort`
-- fresh (results printed as addresses); `comparing` fixed its key to `Int`
-- (a Double key was a type error); a partially applied `on` crashed.
import Data.Char (isDigit)
import Data.Function (on)
import Data.List (groupBy, sort, sortBy, sortOn)
import qualified Data.List as L
import Data.Ord (comparing)

data Shape = Circle Double | Rect Double Double deriving (Show, Eq, Ord)

data Level = Lo | Mid | Hi deriving (Show, Eq, Ord)

area :: Shape -> Double
area (Circle r) = 3 * r * r
area (Rect w h) = w * h

main :: IO ()
main = do
  let shapes = [Rect 2 3, Circle 1, Rect 1 1]
  print (sort [Hi, Lo, Mid])
  print (L.sort [Mid, Hi, Lo])
  print (sort shapes)
  print (maximum shapes, minimum shapes)
  print (maximum [Lo, Hi, Mid], minimum [Hi, Mid])
  print (sortBy (comparing area) shapes)
  print (sortOn area shapes)
  print (sortBy (comparing length) ["ccc", "a", "bb"])
  print (sortBy (compare `on` negate) [3, 1, 2 :: Int])
  print (groupBy ((==) `on` isDigit) "ab12cd345")
  print (((==) `on` isDigit) 'a' '1')
