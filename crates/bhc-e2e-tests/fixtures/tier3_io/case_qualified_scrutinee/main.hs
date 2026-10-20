-- Test: a case on a qualified function's result
-- Verifies: the simplifier does not take `Data.Text.length t` (a qualified
-- name whose module starts uppercase) for a constructor application and
-- fold the case to its default alternative.

import qualified Data.Text as T

markers :: T.Text -> String
markers cs = case T.length cs of
  3 -> "three"
  2 -> "two"
  1 -> "one"
  _ -> "other"

main :: IO ()
main = do
  putStrLn (markers (T.pack "*"))
  putStrLn (markers (T.pack "**"))
  putStrLn (markers (T.pack "***"))
  putStrLn (markers (T.pack "****"))
