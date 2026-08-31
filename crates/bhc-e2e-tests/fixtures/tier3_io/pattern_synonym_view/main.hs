{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ViewPatterns #-}
-- Substituting a pattern synonym's arguments has to reach inside a VIEW's
-- result, or the argument stays the synonym's own variable and nothing has
-- bound it. `Con {}` also names a synonym and matches it whatever its
-- arguments are — pandoc's `Writers.Texinfo` asks "is this a figure?" that
-- way, and `Writers.MediaWiki` matches the arguments out.
module Main where

isFigureTarget :: String -> Maybe String
isFigureTarget t = if take 4 t == "fig:" then Just (drop 4 t) else Nothing

pattern Figure :: String -> String
pattern Figure tgt <- (isFigureTarget -> Just tgt)

describe :: String -> String
describe (Figure t) = "figure " ++ t
describe other = "plain " ++ other

isFig :: String -> Bool
isFig (Figure {}) = True
isFig _ = False

main :: IO ()
main = do
  putStrLn (describe "fig:t")
  putStrLn (describe "other")
  print (isFig "fig:x")
  print (isFig "y")
