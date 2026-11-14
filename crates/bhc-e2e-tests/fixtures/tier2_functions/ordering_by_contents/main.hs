-- Ordering on values that are compared by their contents: strings and
-- lists (lexicographic), tuples, Text, Integer, Double. Equality on these
-- was already by contents; `<`, `compare`, `sort`, `maximum` and `sortOn`
-- compared addresses or the first word, so `"b" < "c"` was False and
-- `sort ["pear", "apple", "fig"]` left the list as it was.
import Data.List (sort, sortBy, maximumBy, minimumBy, sortOn)
import Data.Ord (comparing)
import qualified Data.Text as T

data P = P String Int deriving (Show, Eq, Ord)

main :: IO ()
main = do
  print ("b" < "c", "b" < "a", "b" <= "b", "abc" > "abd", "b" >= "a", "" < "a", "a" < "")
  print (compare "b" "c", compare "b" "a", compare "x" "x", compare "ab" "abc")
  print (max "apple" "banana", min "apple" "banana", sort ["pear", "apple", "fig"])
  print ([1, 2, 3] < [1, 2, 4 :: Int], [1, 2] < [1, 2, 0 :: Int], [2] > [1, 9 :: Int], compare [1, 2] [1, 3 :: Int])
  let name = "mango" in putStrLn (if name < "n" then "first half" else "second half")
  print ((1 :: Int, "b") < (1, "c"), ("a", 2 :: Int) < ("a", 1), ("a", 1 :: Int) <= ("a", 1), (2 :: Int, 'x') > (1, 'z'))
  print (sort [(2 :: Int, "b"), (1, "z"), (2, "a")], sort [[3, 1], [2], [3, 0 :: Int]])
  print (sort [2.5, -1.5, 0, -2.5 :: Double], maximum [-1.5, -2.5 :: Double], minimum [-1.5, -2.5 :: Double])
  print (maximum ["pear", "apple", "fig"], minimum ["pear", "apple", "fig"], maximum [(1 :: Int, "a"), (1, "b")])
  print (sort (map T.pack ["pear", "apple", "fig"]), T.pack "a" < T.pack "b", compare (T.pack "b") (T.pack "a"))
  print (sortBy (comparing snd) [(1 :: Int, "b"), (2, "a")], sortOn fst [("b", 1 :: Int), ("a", 2)])
  print (sort [P "b" 1, P "a" 2, P "a" 1], P "a" 1 < P "a" 2, compare (P "b" 0) (P "a" 9))
  print (sort [3, 1, 2 :: Int], sort "hello", [1.5, 2.5] < [1.5, 3 :: Double], sort [10 ^ 20, 5, 10 ^ 19 :: Integer])
