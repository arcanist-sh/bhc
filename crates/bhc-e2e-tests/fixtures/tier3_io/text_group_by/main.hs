-- Test: Data.Text.groupBy / Data.Text.group (saturated and partial)

{-# LANGUAGE OverloadedStrings #-}
import qualified Data.Text as T
import Data.Char (isSpace)

main :: IO ()
main = do
  let t = T.pack "aa  bb c"
  mapM_ (\g -> putStrLn ("[" ++ T.unpack g ++ "]")) (T.groupBy (\a b -> isSpace a == isSpace b) t)
  print (length (T.group (T.pack "aabccc")))
  let f = T.groupBy (==)
  print (length (f (T.pack "xxyz")))
