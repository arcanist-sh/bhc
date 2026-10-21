-- lookup compares keys with their Eq instance, not by address: String, Text,
-- constructor and tuple keys, beside a scalar Int key.
module Main where
import qualified Data.Text as T

data Color = Red | Green | Blue deriving (Eq, Show)

strTable :: [(String, Int)]
strTable = [("amp;", 1), ("lt;", 2)]

textTable :: [(T.Text, String)]
textTable = [(T.pack "a", "A"), (T.pack "b", "B")]

conTable :: [(Color, String)]
conTable = [(Red, "r"), (Blue, "b")]

intTable :: [(Int, String)]
intTable = [(1, "one"), (2, "two")]

main :: IO ()
main = do
  print (lookup "lt;" strTable)
  print (lookup ("am" ++ "p;") strTable)
  print (lookup "gt;" strTable)
  print (lookup (T.pack "b") textTable)
  print (lookup Blue conTable)
  print (lookup Green conTable)
  print (lookup (1 + 1) intTable)
  print (lookup ((1, 'x') :: (Int, Char)) [((1, 'x'), True)])
