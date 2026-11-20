-- List functions that compare elements for equality, at element types whose
-- machine words do not decide it: strings and types with a derived Eq. They
-- compared addresses (`nub ["a", "b", "a"]` kept both "a"s), and a derived Eq
-- compared a String field by address (`P 1 "a" == P 1 "a"` was False).
import Data.List (delete, group, intersect, nub, union)

data Level = Lo | Hi deriving (Show, Eq, Ord)

data P = P Int String deriving (Show, Eq)

main :: IO ()
main = do
  print (nub [3, 1, 3, 2, 1 :: Int])
  print (nub ["a", "b", "a"])
  print (nub (words "x y x"))
  print (nub [Lo, Hi, Lo])
  print (P 1 "a" == P 1 "a", P 1 "a" == P 1 "b")
  print (nub [P 1 "a", P 2 "b", P 1 "a"])
  print (group [Lo, Lo, Hi])
  print (group (words "a a b"))
  print (Lo `elem` [Hi, Lo], P 1 "a" `elem` [P 1 "a"], "b" `elem` words "a b")
  print (Hi `notElem` [Lo], "q" `notElem` words "a b")
  print (delete Lo [Hi, Lo, Lo], delete "a" (words "b a a"))
  print (union [Lo] [Hi, Lo], intersect [Lo, Hi] [Hi])
  print (union (words "a b") (words "b c"), intersect (words "a b c") (words "c a"))
