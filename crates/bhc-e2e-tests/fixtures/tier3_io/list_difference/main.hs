-- Test: Data.List.(\\) and comparing a lazily filtered list
-- Verifies: (\\) removes elements by value (Text included), and an
-- unevaluated filter result compares equal to an equal list (its cell was
-- read before being forced).

import Data.List ((\\))
import qualified Data.Text as T

type Attr = (T.Text, [T.Text], [(T.Text, T.Text)])

strip :: Attr -> Attr -> [T.Text]
strip (_, cs, _) (_, hcs, _) = cs \\ hcs

keep :: (T.Text, T.Text) -> Bool
keep (k, _) = k /= T.pack "role"

yn :: Bool -> String
yn b = if b then "True" else "False"

main :: IO ()
main = do
  print ([1, 2, 3, 4 :: Int] \\ [2, 4])
  putStrLn (unwords (map T.unpack ([T.pack "a", T.pack "b", T.pack "c"] \\ [T.pack "b"])))
  putStrLn (unwords (map T.unpack (strip (T.empty, [T.pack "x", T.pack "y"], []) (T.empty, [T.pack "x"], []))))
  let kvs = [] :: [(T.Text, T.Text)]
  putStrLn (yn (filter keep kvs /= kvs))
