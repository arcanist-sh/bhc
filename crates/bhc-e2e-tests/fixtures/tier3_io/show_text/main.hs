-- A Text is an opaque RTS handle, so `show` on one printed its ADDRESS —
-- directly, and as the element of a Maybe.
module Main where
import qualified Data.Text as T

label :: T.Text -> String
label t = show t

main :: IO ()
main = do
  print (T.pack "abc")
  putStrLn (label (T.pack "hi"))
  print (T.stripPrefix (T.pack "ab") (T.pack "abc"))
  print (T.stripPrefix (T.pack "zz") (T.pack "abc"))
  print (T.toUpper (T.pack "abc"))
  print (T.words (T.pack "a b"))
