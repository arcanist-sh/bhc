-- A small real Haskell idiom: a Roman numeral converter both ways.
-- Demonstrates: lists, pattern matching, Maybe for partial functions,
-- recursion over a static table, prefix matching.
--
--   bhc demos/03-roman.hs -o /tmp/roman
--   /tmp/roman
--
-- Output:
--   1999 -> MCMXCIX
--   2026 -> MMXXVI
--   MCMXCIX -> Just 1999
--   MMXXVI  -> Just 2026
--   ABCDE   -> Nothing
module Main where

import Data.List (isPrefixOf)

romanTable :: [(Int, String)]
romanTable =
  [ (1000, "M"), (900, "CM"), (500, "D"), (400, "CD")
  , (100,  "C"), ( 90, "XC"), ( 50, "L"), ( 40, "XL")
  , (10,   "X"), (  9, "IX"), (  5, "V"), (  4, "IV")
  , (1,    "I")
  ]

-- Encode: greedy subtraction against the value table.
encode :: Int -> [(Int, String)] -> String
encode _ [] = ""
encode k ((v, s):rest)
  | k <= 0    = ""
  | k >= v    = s ++ encode (k - v) ((v, s):rest)
  | otherwise = encode k rest

toRoman :: Int -> String
toRoman n = encode n romanTable

-- Decode: match prefixes against the same table, longest first.
-- Returns Nothing if any portion fails to decode.
decode :: String -> [(Int, String)] -> Maybe Int
decode []  _  = Just 0
decode _   [] = Nothing
decode str ((v, sym):rest)
  | sym `isPrefixOf` str =
      case decode (drop (length sym) str) ((v, sym):rest) of
        Just n  -> Just (v + n)
        Nothing -> Nothing
  | otherwise = decode str rest

fromRoman :: String -> Maybe Int
fromRoman s = decode s romanTable

main :: IO ()
main = do
  putStrLn ("1999 -> " ++ toRoman 1999)
  putStrLn ("2026 -> " ++ toRoman 2026)
  putStrLn ("MCMXCIX -> " ++ show (fromRoman "MCMXCIX"))
  putStrLn ("MMXXVI  -> " ++ show (fromRoman "MMXXVI"))
  putStrLn ("ABCDE   -> " ++ show (fromRoman "ABCDE"))
