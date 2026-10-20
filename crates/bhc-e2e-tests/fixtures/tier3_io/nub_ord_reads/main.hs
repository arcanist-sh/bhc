-- Test: Data.Containers.ListUtils.nubOrd and reads at Int
-- Verifies: nubOrd keeps first occurrences in order; reads parses a leading
-- (optionally signed) decimal and returns the rest of the input, and
-- answers [] when there is no number.

import Data.Char (isSpace)
import Data.Containers.ListUtils (nubOrd)

safeStrRead :: String -> Maybe Int
safeStrRead s = case reads s of
  (d, x) : _ | all isSpace x -> Just d
  _ -> Nothing

showRead :: Maybe Int -> String
showRead = maybe "Nothing" show

main :: IO ()
main = do
  print (nubOrd [3, 1, 3, 2, 1 :: Int])
  putStrLn (showRead (safeStrRead "42"))
  putStrLn (showRead (safeStrRead " -7 "))
  putStrLn (showRead (safeStrRead "x1"))
  putStrLn (showRead (safeStrRead "12ab"))
  case reads "15 rest" :: [(Int, String)] of
    [(n, r)] -> putStrLn (show n ++ "|" ++ r)
    _ -> putStrLn "none"
