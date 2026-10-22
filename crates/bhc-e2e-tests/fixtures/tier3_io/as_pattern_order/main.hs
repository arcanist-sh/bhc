-- An as-pattern over a refutable pattern, before a constructor clause,
-- keeps source order: in a top-level function, a local function and a case
-- expression, with and without a guard (a tag tokenizer's text scan).
module Main where

import Data.Char (isAlpha)

topLevel :: String -> (String, String)
topLevel [] = ([], [])
topLevel inp@('<' : _) = ([], inp)
topLevel (x : rest) = let (a, b) = topLevel rest in (x : a, b)

local :: String -> (String, String)
local = go
  where
    go [] = ([], [])
    go inp@('<' : _) = ([], inp)
    go (x : rest) = let (a, b) = go rest in (x : a, b)

guarded :: String -> (String, String)
guarded = go
  where
    go [] = ([], [])
    go inp@('<' : rest)
      | startsTag rest = ([], inp)
    go (x : rest) = let (a, b) = go rest in (x : a, b)
    startsTag ('/' : n : _) = isAlpha n
    startsTag (n : _) = isAlpha n
    startsTag [] = False

viaCase :: String -> String
viaCase s = case s of
  [] -> "empty"
  whole@('<' : _) -> "tag " ++ whole
  (c : _) -> "char " ++ [c]

main :: IO ()
main = do
  print (topLevel "ab<cd")
  print (local "ab<cd")
  print (guarded "t&lt; x</p><br>")
  print (guarded "a < b")
  putStrLn (viaCase "<p>")
  putStrLn (viaCase "xy")
  putStrLn (viaCase "")
