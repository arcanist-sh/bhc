-- `Data.List.union` keeps the first list and appends the elements of the
-- second that are not in it, without duplicates. It was a plain append
-- (issue #27).
import Data.List (union)

main :: IO ()
main = do
  print (union [1,2] [2,3::Int])
  print (union [1,2,2] [3,3,1,4::Int])
  putStrLn (union "abc" "bcd")
  print (length (union ([] :: [Int]) [1,1,2]))
