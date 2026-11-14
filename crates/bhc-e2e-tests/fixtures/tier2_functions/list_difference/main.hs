-- `xs \\ ys` removes, for each element of `ys`, the FIRST equal element of
-- `xs`, comparing by the element type: it removed every occurrence, and
-- compared enum constructors by address.
import Data.List ((\\))
import qualified Data.Text as T

data Level = Lo | Hi deriving (Show, Eq)

main :: IO ()
main = do
  print ([Lo, Hi, Lo] \\ [Lo])
  print ([1, 2, 3, 2 :: Int] \\ [2])
  print (words "a b a" \\ ["a"])
  print (map T.unpack ([T.pack "a", T.pack "b", T.pack "a"] \\ [T.pack "a"]))
  print ([(1, 'a'), (2, 'b')] \\ [(1 :: Int, 'a')])
