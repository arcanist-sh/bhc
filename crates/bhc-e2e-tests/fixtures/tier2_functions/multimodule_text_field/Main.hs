{-# LANGUAGE OverloadedStrings #-}
-- OverloadedStrings literals in an IMPORTED constructor's `Text` fields:
-- positional, nested, and record-syntax construction.
module Main where
import qualified Data.Text as T
import Defs

render :: Inl -> T.Text
render (Str t) = t
render (Emph is) = T.concat ["*", T.concat (map render is), "*"]
render Space = " "

main :: IO ()
main = do
  print (T.length (render (Str "hi")))
  putStrLn (T.unpack (T.concat (map render [Str "a", Space, Emph [Str "b"]])))
  let d = Doc { docTitle = "Title", docBody = [Str "x"] }
  putStrLn (T.unpack (T.toUpper (docTitle d)))
  print (T.any (== 'i') (render (Str "hi")))
