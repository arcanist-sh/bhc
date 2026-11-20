-- `read`, `readMaybe` and `reads` at derived data types, records, tuples,
-- nested lists, Maybe and Either (#105).
module Main where

import Text.Read (readMaybe)

data Color = Red | Green | Blue
  deriving (Show, Read, Eq)

data P = P { name :: String, age :: Int }
  deriving (Show, Read)

data Shape = Circle Double | Rect Double Double | Unit
  deriving (Show, Read)

data Tree a = Leaf | Node (Tree a) a (Tree a)
  deriving (Show, Read)

newtype Age = Age Int
  deriving (Show, Read)

main :: IO ()
main = do
  print (read "Blue" :: Color)
  print (read "P {name = \"a\", age = 3}" :: P)
  print (read "[(1,\"a\"),(2,\"b\")]" :: [(Int, String)])
  print (read " ( Green ) " :: Color)
  print (map read ["Red", "Blue"] :: [Color])
  print (read "[Circle 1.5,Rect 2.0 3.0,Unit]" :: [Shape])
  print (read "Node (Node Leaf 1 Leaf) 2 Leaf" :: Tree Int)
  print (read "Node Leaf \"x\" Leaf" :: Tree String)
  print (read "Age 41" :: Age)
  print (read "(1,True,'c',\"s\",2.5)" :: (Int, Bool, Char, String, Double))
  print (read "Just (-3)" :: Maybe Int)
  print (read "[Left 1,Right \"r\"]" :: [Either Int String])
  print (read "[[1,2],[3]]" :: [[Int]])
  print (read "123456789012345678901234567890" :: Integer)
  print (read "()" :: ())
  print (read "[LT,GT]" :: [Ordering])
  print (readMaybe "Green" :: Maybe Color)
  print (readMaybe "Purple" :: Maybe Color)
  print (readMaybe "P {name = \"b\", age = 4}" :: Maybe P)
  print (readMaybe "P {name = \"b\"}" :: Maybe P)
  print (readMaybe "(1,2)" :: Maybe (Int, Int))
  print (readMaybe "(1,2" :: Maybe (Int, Int))
  print (reads "Blue rest" :: [(Color, String)])
  print (reads "Mauve" :: [(Color, String)])
  print (reads "(1,2) tail" :: [((Int, Int), String)])
  print (read "42" :: Int, read "2.5" :: Double, read "[1,2,3]" :: [Int])
  print (fmap age (readMaybe "P {name = \"c\", age = 9}"))
  print ((read "Red" :: Color) == Red)
