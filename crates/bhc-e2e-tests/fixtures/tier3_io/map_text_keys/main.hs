-- A Map orders its keys as machine words, so a boxed key compared by ADDRESS
-- and two equal Texts never matched: every `Map Text v` in pandoc answered
-- Nothing.
module Main where
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.Text as T

main :: IO ()
main = do
  let ms = M.fromList [("k" :: String, 7 :: Int), ("j", 8)]
  print (M.lookup ("k" :: String) ms)
  print (M.findWithDefault (0 :: Int) ("j" :: String) ms)
  print (M.member ("z" :: String) ms)
  let mt = M.fromList [(T.pack "k", 7 :: Int)]
  print (M.findWithDefault (0 :: Int) (T.pack "k") mt)
  print (M.member (T.pack "k") mt)
  print (M.findWithDefault (0 :: Int) (T.pack "z") mt)
  let mi = M.fromList [(1 :: Int, 10 :: Int), (2, 20)]
  print (M.lookup (2 :: Int) mi)
  let st = S.fromList [T.pack "a", T.pack "b"]
  print (S.member (T.pack "a") st)
  print (S.member (T.pack "c") st)
