-- Test: list (in)equality compares elements by contents
-- Verifies: lists of Text and of Text pairs compare their elements by value,
-- not by pointer (pandoc's HTML writer compared attribute lists this way
-- and wrapped every header in a <section>).

import qualified Data.Text as T

yn :: Bool -> String
yn b = if b then "True" else "False"

sameAttrs :: [T.Text] -> [T.Text] -> [(T.Text, T.Text)] -> [(T.Text, T.Text)] -> String
sameAttrs hc dc hk dk = yn (hc /= dc) ++ " " ++ yn (hk /= dk)

main :: IO ()
main = do
  putStrLn (yn ([T.pack "a"] /= [T.pack "a"]))
  putStrLn (yn ([T.pack "a"] == [T.pack "b"]))
  putStrLn (yn ([(T.pack "a", T.pack "b")] == [(T.pack "a", T.pack "b")]))
  putStrLn (sameAttrs [T.pack "x"] [T.pack "x"] [(T.pack "k", T.pack "v")] [(T.pack "k", T.pack "v")])
  putStrLn (sameAttrs [T.pack "x"] [T.pack "y"] [] [(T.pack "k", T.pack "v")])
