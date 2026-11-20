-- Two maps, or two sets, are equal when they hold the same entries. `==` used
-- to compare the two objects' addresses, so equal maps built separately were
-- unequal.
import qualified Data.Map as M
import qualified Data.Set as S
data Color = Red | Green deriving (Show, Eq, Ord)
main :: IO ()
main = do
  let a = M.fromList [(1 :: Int, "one"), (2, "two")]
      b = M.insert 2 "two" (M.fromList [(1, "one")])
      c = M.insert 3 "three" a
  print (a == b, a == c, a /= c, a /= b)
  print (M.fromList [(1 :: Int, 10 :: Int)] == M.fromList [(1, 10)], M.fromList [(1 :: Int, 10 :: Int)] == M.fromList [(1, 11)])
  print (M.fromList [("k", Red)] == M.fromList [("k", Red)], M.fromList [("k", Red)] == M.fromList [("k", Green)])
  print (S.fromList [3, 1, 2 :: Int] == S.fromList [1, 2, 3], S.fromList "abc" == S.fromList "abd", S.fromList ["x", "y"] == S.fromList ["y", "x"])
  print (M.empty == (M.empty :: M.Map Int Int), S.empty == (S.empty :: S.Set Int))
