module Main where
import qualified Data.Text as T

nvs :: Char -> Bool
nvs 'z' = True
nvs _ = False

isS :: Char -> Bool
isS 'x' = True
isS c = nvs c

main :: IO ()
main = do
  print (if T.any isS (T.pack "abc") then 1 else 0 :: Int)
  print (if T.any isS (T.pack "abz") then 1 else 0 :: Int)
  print (if T.all isS (T.pack "xz") then 1 else 0 :: Int)
  print (if any isS "abc" then 1 else 0 :: Int)
