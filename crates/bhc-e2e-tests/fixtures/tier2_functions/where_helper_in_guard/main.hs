module Main where
import qualified Data.Text as T

f :: T.Text -> Int
f t
  | T.any isS t = 1
  | otherwise = 0
  where
    isS 'x' = True
    isS c = c == 'y'

g :: String -> Int
g s
  | any isS s = 1
  | otherwise = 0
  where
    isS 'x' = True
    isS c = c == 'y'

main :: IO ()
main = do
  print (f (T.pack "abx"))
  print (f (T.pack "abc"))
  print (g "aby")
