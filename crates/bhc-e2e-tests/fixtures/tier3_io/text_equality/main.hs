-- `Text` is an opaque RTS handle; `==` fell through to the pointer path and
-- answered False for every pair of equal Texts.
module Main where
import qualified Data.Text as T

eqt :: T.Text -> T.Text -> Bool
eqt a b = a == b

main :: IO ()
main = do
  print (T.pack "abc" == T.pack "abc")
  print (T.pack "abc" == T.pack "abd")
  print (eqt (T.pack "abc") (T.pack "abc"))
  print (eqt (T.pack "abc") (T.pack "abd"))
  print (T.pack "abc" /= T.pack "abd")
