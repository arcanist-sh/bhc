-- elem/notElem compare elements by their Eq, not by address.
{-# LANGUAGE OverloadedStrings #-}
import qualified Data.Text as T
import Data.Text (Text)

data Style = DefaultStyle | Decimal | Example deriving (Eq, Show)
data P = P Int Style deriving Eq

main :: IO ()
main = do
  print (DefaultStyle `elem` [DefaultStyle, Decimal, Example])
  print (Example `elem` [DefaultStyle, Decimal])
  print (Example `notElem` [DefaultStyle, Decimal])
  print (P 1 Decimal `elem` [P 2 Decimal, P 1 Decimal])
  print (("a" :: Text) `elem` ["b", "a"])
  print ((1 :: Int, 'x') `elem` [(1, 'x')])
  print ((10 :: Integer) `elem` [5, 10])
  print ((3 :: Int) `elem` [1, 2, 3])
  print ("ab" `elem` ["x", "ab"])
