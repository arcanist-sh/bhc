-- A small real Haskell idiom: a Roman numeral converter both ways.
-- Demonstrates: ADTs, pattern matching on integer ranges, Maybe for
-- partial functions, list-based recursion, lookup over a static table.
--
--   bhc demos/03-roman.hs -o /tmp/roman
--   /tmp/roman
--
-- Output:
--   1999 -> MCMXCIX
--   2026 -> MMXXVI
--   MCMXCIX -> Just 1999
--   MMXXVI  -> Just 2026
--   nonsense -> Nothing
module Main where

import Data.List (isPrefixOf)

-- Encode: greedy subtraction against a value table.
romanTable :: [(Int, String)]
romanTable =
  [ (1000, "M"), (900, "CM"), (500, "D"), (400, "CD")
  , (100,  "C"), ( 90, "XC"), ( 50, "L"), ( 40, "XL")
  , (10,   "X"), (  9, "IX"), (  5, "V"), (  4, "IV")
  , (1,    "I")
  ]

toRoman :: Int -> String
toRoman n
  | n <= 0    = ""
  | otherwise = go n romanTable
  where
    go _ [] = ""
    go k ((v, s):rest)
      | k >= v    = s ++ go (k - v) ((v, s):rest)
      | otherwise = go k rest

-- Decode: match prefixes against the same table, longest first.
-- Returns Nothing if any portion fails to decode.
fromRoman :: String -> Maybe Int
fromRoman s = go s 0
  where
    go []  acc = Just acc
    go str acc =
      case dropWhile (\(_, sym) -> not (sym `isPrefixOf` str)) romanTable of
        []           -> Nothing
        (v, sym):_   -> go (drop (length sym) str) (acc + v)

main :: IO ()
main = do
  putStrLn ("1999 -> " ++ toRoman 1999)
  putStrLn ("2026 -> " ++ toRoman 2026)
  putStrLn ("MCMXCIX -> " ++ show (fromRoman "MCMXCIX"))
  putStrLn ("MMXXVI  -> " ++ show (fromRoman "MMXXVI"))
  putStrLn ("nonsense -> " ++ show (fromRoman "ABCDE"))
