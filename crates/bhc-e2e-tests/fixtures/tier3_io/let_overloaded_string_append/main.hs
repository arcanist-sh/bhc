{-# LANGUAGE OverloadedStrings #-}
-- A local binding with no signature built from string literals and `<>`,
-- used at a user `IsString` type. The binding was generalized, its use
-- instantiated a fresh copy, and the literals defaulted to `String`: the
-- value was built with list append and read as the user type.
import Data.String (IsString (..))
import Data.Char (toUpper)

newtype V = V String deriving Show

instance IsString V where
  fromString = V . map toUpper

instance Semigroup V where
  V a <> V b = V (a ++ "+" ++ b)

use :: V -> String
use (V s) = s

twoLiterals :: String
twoLiterals = let c = "pre " <> "x" in use c

fromCase :: Int -> String
fromCase n =
  let a = case n of
        0 -> "zero"
        _ -> "other"
      c = "pre " <> a
  in use c

inDo :: Int -> IO String
inDo n = do
  let a = if n > 0 then "pos" else "neg"
  let c = "sign " <> a
  return (use c)

main :: IO ()
main = do
  putStrLn twoLiterals
  putStrLn (fromCase 0)
  putStrLn (fromCase 1)
  inDo 3 >>= putStrLn
  print (let v = "a" <> "b" <> "c" in v :: V)
