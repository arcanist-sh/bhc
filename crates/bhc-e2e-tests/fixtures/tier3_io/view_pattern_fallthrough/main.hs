-- A view pattern whose result fails to match, at its head or in a nested
-- sub-pattern, falls through to the next equation or alternative
-- (pandoc's scanForPipe: `(_, T.uncons -> Just ('|', _)) -> …; _ -> mzero`).
{-# LANGUAGE ViewPatterns #-}
module Main where
import qualified Data.Text as T

a :: T.Text -> Bool
a t = case T.uncons t of
  Just ('|', _) -> True
  _ -> False

b :: T.Text -> Bool
b (T.uncons -> Just ('|', _)) = True
b _ = False

c :: (Int, T.Text) -> Bool
c p = case p of
  (_, T.uncons -> Just ('|', _)) -> True
  _ -> False

d :: (Int, T.Text) -> Bool
d p = case p of
  (_, T.uncons -> Just _) -> True
  _ -> False

main :: IO ()
main = do
  print (a (T.pack "|x"), a (T.pack "x"))
  print (a T.empty)
  print (b (T.pack "|x"))
  print (b (T.pack "x"))
  print (b T.empty)
  print (c (1, T.pack "|x"))
  print (c (1, T.pack "x"))
  print (d (1, T.empty))
  print (c (1, T.empty))
