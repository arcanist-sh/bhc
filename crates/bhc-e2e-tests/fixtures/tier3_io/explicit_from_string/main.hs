{-# LANGUAGE OverloadedStrings #-}
-- An explicit `fromString` at a type other than `String` is that type's own
-- conversion, in ordinary code and inside another `IsString` instance. It was
-- the identity, so a char list stood in for the value.
import Data.String (IsString (..))
import Data.Char (toUpper)
import qualified Data.Text as T

data Choice = Static String | Dyn String | Both Choice Choice
  deriving Show

instance IsString Choice where
  fromString = Dyn . map toUpper

newtype Value = Value Choice
  deriving Show

instance IsString Value where
  fromString = Value . fromString

attr :: Value -> String
attr (Value c) = "id=" ++ show c

named :: Int -> String
named n =
  let idName = if n <= 0 then "footnotes" else "footnotes-" <> show (n + 1)
  in attr (fromString idName)

main :: IO ()
main = do
  putStrLn (attr "literal")
  putStrLn (attr (fromString ("explicit" :: String)))
  putStrLn (named 0)
  putStrLn (named 2)
  print (fromString "x" :: Choice)
  let name = "abc" ++ show (4 :: Int)
  print (T.length (fromString name), T.toUpper (fromString name))
