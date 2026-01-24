{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeSynonymInstances #-}

-- |
-- Module      : BHC.Data.String
-- Description : String utilities and conversions
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
--
-- This module provides utilities for working with 'String' values.
-- In BHC (as in GHC), 'String' is a type synonym for @[Char]@.
--
-- = Note on Performance
--
-- 'String' is a linked list of characters and is not efficient for
-- large text processing. For performance-critical text handling,
-- consider using @BHC.Data.Text@ instead.
--
-- = Usage
--
-- @
-- import BHC.Data.String
--
-- -- Using IsString for overloaded string literals
-- {-# LANGUAGE OverloadedStrings #-}
--
-- myText :: String
-- myText = \"Hello, World!\"
--
-- -- lines and words
-- lines \"hello\\nworld\" == [\"hello\", \"world\"]
-- words \"hello world\" == [\"hello\", \"world\"]
-- @

module BHC.Data.String
    ( -- * The String type
      String

      -- * IsString class
    , IsString(..)

      -- * Basic functions
    , lines
    , words
    , unlines
    , unwords

      -- * Conversion
    , showString
    , showChar
    , readMaybe
    , readEither

      -- * Predicates
    , isSpace
    , isAlpha
    , isAlphaNum
    , isDigit
    , isLower
    , isUpper
    , isPunctuation
    , isControl
    , isPrint
    , isAscii
    , isLatin1

      -- * Case conversion
    , toLower
    , toUpper
    , toTitle

      -- * String utilities
    , strip
    , stripStart
    , stripEnd
    , splitOn
    , replace
    , intercalate
    , intersperse
    , isPrefixOf
    , isSuffixOf
    , isInfixOf

      -- * Justification
    , justifyLeft
    , justifyRight
    , center

      -- * Escaping
    , escape
    , unescape
    ) where

import Prelude hiding (String, lines, words, unlines, unwords)
import qualified Prelude

import Data.Char (isSpace, isAlpha, isAlphaNum, isDigit, isLower, isUpper,
                  isPunctuation, isControl, isPrint, isAscii, isLatin1,
                  toLower, toUpper, toTitle, chr, ord)
import Data.List (isPrefixOf, isSuffixOf, isInfixOf, intercalate, intersperse)
import Text.Read (readMaybe, readEither)

-- | A 'String' is a list of characters.
type String = Prelude.String

-- ============================================================
-- IsString class
-- ============================================================

-- | Class for types that can be constructed from a string literal.
-- Used with the OverloadedStrings extension.
class IsString a where
    -- | Convert a 'String' to the type @a@.
    fromString :: String -> a

instance IsString String where
    fromString = id

instance IsString [Char] where
    fromString = id

-- ============================================================
-- Basic functions
-- ============================================================

-- | Break a string into a list of lines, splitting on newlines.
--
-- >>> lines "hello\nworld"
-- ["hello","world"]
--
-- >>> lines "hello\nworld\n"
-- ["hello","world"]
--
-- >>> lines ""
-- []
lines :: String -> [String]
lines "" = []
lines s = case break (== '\n') s of
    (l, s') -> l : case s' of
        [] -> []
        _:s'' -> lines s''

-- | Break a string into a list of words, splitting on whitespace.
--
-- >>> words "hello world"
-- ["hello","world"]
--
-- >>> words "  hello   world  "
-- ["hello","world"]
--
-- >>> words ""
-- []
words :: String -> [String]
words s = case dropWhile isSpace s of
    "" -> []
    s' -> w : words s''
      where (w, s'') = break isSpace s'

-- | Join lines with newline characters.
--
-- >>> unlines ["hello", "world"]
-- "hello\nworld\n"
unlines :: [String] -> String
unlines = concatMap (++ "\n")

-- | Join words with spaces.
--
-- >>> unwords ["hello", "world"]
-- "hello world"
unwords :: [String] -> String
unwords [] = ""
unwords ws = foldr1 (\w s -> w ++ ' ' : s) ws

-- ============================================================
-- Show utilities
-- ============================================================

-- | Prepend a string to a ShowS function.
--
-- >>> showString "hello" " world"
-- "hello world"
showString :: String -> ShowS
showString = (++)

-- | Prepend a character to a ShowS function.
--
-- >>> showChar 'x' "yz"
-- "xyz"
showChar :: Char -> ShowS
showChar = (:)

-- ============================================================
-- String utilities
-- ============================================================

-- | Remove leading and trailing whitespace.
--
-- >>> strip "  hello world  "
-- "hello world"
strip :: String -> String
strip = stripEnd . stripStart

-- | Remove leading whitespace.
--
-- >>> stripStart "  hello"
-- "hello"
stripStart :: String -> String
stripStart = dropWhile isSpace

-- | Remove trailing whitespace.
--
-- >>> stripEnd "hello  "
-- "hello"
stripEnd :: String -> String
stripEnd = reverse . dropWhile isSpace . reverse

-- | Split a string on a delimiter.
--
-- >>> splitOn "," "a,b,c"
-- ["a","b","c"]
--
-- >>> splitOn "," ",,"
-- ["","",""]
--
-- >>> splitOn "," ""
-- [""]
splitOn :: String -> String -> [String]
splitOn _ "" = [""]
splitOn sep s
    | null sep = error "splitOn: empty delimiter"
    | otherwise = go s
  where
    n = length sep
    go str = case findIndex' (isPrefixOf sep) (tails str) of
        Nothing -> [str]
        Just i -> take i str : go (drop (i + n) str)

-- | Replace all occurrences of a substring.
--
-- >>> replace "world" "there" "hello world"
-- "hello there"
--
-- >>> replace "o" "0" "hello world"
-- "hell0 w0rld"
replace :: String -> String -> String -> String
replace _ _ "" = ""
replace old new s
    | null old = error "replace: empty old string"
    | old `isPrefixOf` s = new ++ replace old new (drop (length old) s)
    | otherwise = head s : replace old new (tail s)

-- ============================================================
-- Justification
-- ============================================================

-- | Left-justify a string, padding with a character on the right.
--
-- >>> justifyLeft 10 '.' "hello"
-- "hello....."
justifyLeft :: Int -> Char -> String -> String
justifyLeft n c s
    | len >= n = s
    | otherwise = s ++ replicate (n - len) c
  where len = length s

-- | Right-justify a string, padding with a character on the left.
--
-- >>> justifyRight 10 '.' "hello"
-- ".....hello"
justifyRight :: Int -> Char -> String -> String
justifyRight n c s
    | len >= n = s
    | otherwise = replicate (n - len) c ++ s
  where len = length s

-- | Center a string, padding with a character on both sides.
-- If the padding is uneven, the extra character goes on the right.
--
-- >>> center 10 '.' "hello"
-- "..hello..."
center :: Int -> Char -> String -> String
center n c s
    | len >= n = s
    | otherwise = replicate leftPad c ++ s ++ replicate rightPad c
  where
    len = length s
    total = n - len
    leftPad = total `div` 2
    rightPad = total - leftPad

-- ============================================================
-- Escaping
-- ============================================================

-- | Escape special characters in a string for use in source code.
--
-- >>> escape "hello\nworld"
-- "hello\\nworld"
--
-- >>> escape "tab\there"
-- "tab\\there"
escape :: String -> String
escape = concatMap escapeChar
  where
    escapeChar '\n' = "\\n"
    escapeChar '\t' = "\\t"
    escapeChar '\r' = "\\r"
    escapeChar '\\' = "\\\\"
    escapeChar '"'  = "\\\""
    escapeChar '\'' = "\\'"
    escapeChar c
        | isPrint c = [c]
        | otherwise = "\\x" ++ showHex (ord c)

    showHex n
        | n < 16 = "0" ++ showHex' n
        | otherwise = showHex' n
    showHex' n
        | n < 10 = [chr (n + ord '0')]
        | otherwise = [chr (n - 10 + ord 'a')]

-- | Unescape a string with escape sequences.
--
-- >>> unescape "hello\\nworld"
-- "hello\nworld"
--
-- This function handles common escape sequences:
-- @\\n@, @\\t@, @\\r@, @\\\\@, @\\"@, @\\'@
unescape :: String -> String
unescape "" = ""
unescape ('\\':c:cs) = case c of
    'n'  -> '\n' : unescape cs
    't'  -> '\t' : unescape cs
    'r'  -> '\r' : unescape cs
    '\\' -> '\\' : unescape cs
    '"'  -> '"'  : unescape cs
    '\'' -> '\'' : unescape cs
    '0'  -> '\0' : unescape cs
    'x'  -> case cs of
        (h1:h2:rest) -> chr (hexDigit h1 * 16 + hexDigit h2) : unescape rest
        _ -> '\\' : 'x' : unescape cs
    _    -> c : unescape cs
unescape (c:cs) = c : unescape cs

hexDigit :: Char -> Int
hexDigit c
    | c >= '0' && c <= '9' = ord c - ord '0'
    | c >= 'a' && c <= 'f' = ord c - ord 'a' + 10
    | c >= 'A' && c <= 'F' = ord c - ord 'A' + 10
    | otherwise = 0

-- ============================================================
-- Internal helpers
-- ============================================================

-- | Find the index of the first element satisfying a predicate.
findIndex' :: (a -> Bool) -> [a] -> Maybe Int
findIndex' p xs = go 0 xs
  where
    go _ [] = Nothing
    go !i (y:ys)
        | p y = Just i
        | otherwise = go (i + 1) ys

-- | Get all suffixes of a list.
tails :: [a] -> [[a]]
tails [] = [[]]
tails xs@(_:xs') = xs : tails xs'

-- | Break a list at the first element satisfying a predicate.
break :: (a -> Bool) -> [a] -> ([a], [a])
break _ [] = ([], [])
break p xs@(x:xs')
    | p x = ([], xs)
    | otherwise = let (ys, zs) = break p xs' in (x:ys, zs)

-- | Concatenate a list of lists with a separator.
-- (Re-exported from Data.List but included here for completeness)
-- intercalate :: [a] -> [[a]] -> [a]
-- Already imported from Data.List
