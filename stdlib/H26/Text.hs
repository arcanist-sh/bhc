-- |
-- Module      : H26.Text
-- Description : UTF-8 text handling
-- License     : BSD-3-Clause
--
-- The H26.Text module provides efficient UTF-8 text processing.
-- Text is stored as UTF-8 encoded bytes with O(1) slicing.

{-# HASKELL_EDITION 2026 #-}

module H26.Text
  ( -- * Text Type
    Text

    -- * Construction
  , empty
  , singleton
  , pack
  , unpack
  , fromString
  , toString

    -- * Basic Operations
  , length
  , null
  , cons
  , snoc
  , append
  , head
  , tail
  , last
  , init
  , uncons
  , unsnoc

    -- * Transformations
  , map
  , intercalate
  , intersperse
  , reverse
  , replace
  , toCaseFold
  , toLower
  , toUpper
  , toTitle

    -- * Folds
  , foldl
  , foldl'
  , foldr
  , foldl1
  , foldr1

    -- * Special Folds
  , concat
  , concatMap
  , any
  , all
  , maximum
  , minimum

    -- * Slicing (Views)
  , take
  , takeEnd
  , drop
  , dropEnd
  , takeWhile
  , takeWhileEnd
  , dropWhile
  , dropWhileEnd
  , splitAt
  , span
  , break

    -- * Breaking into Lines and Words
  , lines
  , unlines
  , words
  , unwords

    -- * Splitting
  , split
  , splitOn
  , chunksOf

    -- * Searching
  , elem
  , find
  , filter
  , partition
  , breakOn
  , breakOnEnd
  , breakOnAll

    -- * Indexing
  , index
  , (!?)
  , findIndex

    -- * Predicates
  , isPrefixOf
  , isSuffixOf
  , isInfixOf

    -- * Stripping
  , strip
  , stripStart
  , stripEnd
  , stripPrefix
  , stripSuffix
  , commonPrefixes

    -- * Encoding
  , encodeUtf8
  , decodeUtf8
  , decodeUtf8'
  , decodeUtf8With

    -- * I/O
  , readFile
  , writeFile
  , appendFile
  , interact
  , getLine
  , getContents
  , putStr
  , putStrLn
  , hGetContents
  , hGetLine
  , hPutStr
  , hPutStrLn
  ) where

import Prelude hiding
  ( length, null, head, tail, last, init, map, reverse
  , foldl, foldr, concat, concatMap, any, all, maximum, minimum
  , take, drop, takeWhile, dropWhile, splitAt, span, break
  , lines, unlines, words, unwords, elem, filter
  , readFile, writeFile, appendFile, interact, getLine, getContents
  , putStr, putStrLn
  )

-- | Immutable UTF-8 encoded text.
--
-- Text is stored compactly as UTF-8 bytes. All operations preserve
-- valid UTF-8 encoding. Slicing creates views without copying.
data Text

-- | O(1). The empty text.
empty :: Text

-- | O(1). A single character.
singleton :: Char -> Text

-- | O(n). Pack a list of characters.
pack :: [Char] -> Text

-- | O(n). Unpack to a list of characters.
unpack :: Text -> [Char]

-- | O(n). Convert from String.
fromString :: String -> Text

-- | O(n). Convert to String.
toString :: Text -> String

-- | O(n). Length in characters (not bytes).
length :: Text -> Int

-- | O(1). Test if empty.
null :: Text -> Bool

-- | O(n). Prepend a character.
cons :: Char -> Text -> Text

-- | O(n). Append a character.
snoc :: Text -> Char -> Text

-- | O(n). Append two texts.
append :: Text -> Text -> Text

-- | O(1). First character (unsafe on empty).
head :: Text -> Char

-- | O(1). All but the first character (view).
tail :: Text -> Text

-- | O(n). Last character (unsafe on empty).
last :: Text -> Char

-- | O(1). All but the last character (view).
init :: Text -> Text

-- | O(1). Uncons first character.
uncons :: Text -> Maybe (Char, Text)

-- | O(1). Unsnoc last character.
unsnoc :: Text -> Maybe (Text, Char)

-- | O(n). Map a function over characters.
map :: (Char -> Char) -> Text -> Text

-- | O(n). Intercalate separator between texts.
intercalate :: Text -> [Text] -> Text

-- | O(n). Intersperse character between characters.
intersperse :: Char -> Text -> Text

-- | O(n). Reverse text.
reverse :: Text -> Text

-- | O(n*m). Replace all occurrences.
replace :: Text -> Text -> Text -> Text

-- | O(n). Case fold (for case-insensitive comparison).
toCaseFold :: Text -> Text

-- | O(n). Convert to lowercase.
toLower :: Text -> Text

-- | O(n). Convert to uppercase.
toUpper :: Text -> Text

-- | O(n). Convert to title case.
toTitle :: Text -> Text

-- | O(n). Left fold.
foldl :: (a -> Char -> a) -> a -> Text -> a

-- | O(n). Strict left fold.
foldl' :: (a -> Char -> a) -> a -> Text -> a

-- | O(n). Right fold.
foldr :: (Char -> a -> a) -> a -> Text -> a

-- | O(n). Left fold without starting value.
foldl1 :: (Char -> Char -> Char) -> Text -> Char

-- | O(n). Right fold without starting value.
foldr1 :: (Char -> Char -> Char) -> Text -> Char

-- | O(n). Concatenate texts.
concat :: [Text] -> Text

-- | O(n). Map then concatenate.
concatMap :: (Char -> Text) -> Text -> Text

-- | O(n). Any character satisfies predicate.
any :: (Char -> Bool) -> Text -> Bool

-- | O(n). All characters satisfy predicate.
all :: (Char -> Bool) -> Text -> Bool

-- | O(n). Maximum character.
maximum :: Text -> Char

-- | O(n). Minimum character.
minimum :: Text -> Char

-- | O(n). Take first n characters (view).
take :: Int -> Text -> Text

-- | O(n). Take last n characters (view).
takeEnd :: Int -> Text -> Text

-- | O(n). Drop first n characters (view).
drop :: Int -> Text -> Text

-- | O(n). Drop last n characters (view).
dropEnd :: Int -> Text -> Text

-- | O(n). Take while predicate holds.
takeWhile :: (Char -> Bool) -> Text -> Text

-- | O(n). Take from end while predicate holds.
takeWhileEnd :: (Char -> Bool) -> Text -> Text

-- | O(n). Drop while predicate holds.
dropWhile :: (Char -> Bool) -> Text -> Text

-- | O(n). Drop from end while predicate holds.
dropWhileEnd :: (Char -> Bool) -> Text -> Text

-- | O(n). Split at character position.
splitAt :: Int -> Text -> (Text, Text)

-- | O(n). Span while predicate holds.
span :: (Char -> Bool) -> Text -> (Text, Text)

-- | O(n). Break at first character satisfying predicate.
break :: (Char -> Bool) -> Text -> (Text, Text)

-- | O(n). Break into lines.
lines :: Text -> [Text]

-- | O(n). Join lines.
unlines :: [Text] -> Text

-- | O(n). Break into words.
words :: Text -> [Text]

-- | O(n). Join words with spaces.
unwords :: [Text] -> Text

-- | O(n). Split on predicate.
split :: (Char -> Bool) -> Text -> [Text]

-- | O(n*m). Split on substring.
splitOn :: Text -> Text -> [Text]

-- | O(n). Split into chunks of size n.
chunksOf :: Int -> Text -> [Text]

-- | O(n). Element membership.
elem :: Char -> Text -> Bool

-- | O(n). Find first character satisfying predicate.
find :: (Char -> Bool) -> Text -> Maybe Char

-- | O(n). Filter characters.
filter :: (Char -> Bool) -> Text -> Text

-- | O(n). Partition by predicate.
partition :: (Char -> Bool) -> Text -> (Text, Text)

-- | O(n*m). Break at first occurrence of pattern.
breakOn :: Text -> Text -> (Text, Text)

-- | O(n*m). Break at last occurrence of pattern.
breakOnEnd :: Text -> Text -> (Text, Text)

-- | O(n*m). Find all occurrences of pattern.
breakOnAll :: Text -> Text -> [(Text, Text)]

-- | O(n). Index by character position (unsafe).
index :: Text -> Int -> Char

-- | O(n). Safe indexing.
(!?) :: Text -> Int -> Maybe Char

-- | O(n). Find index of first match.
findIndex :: (Char -> Bool) -> Text -> Maybe Int

-- | O(n). Test prefix.
isPrefixOf :: Text -> Text -> Bool

-- | O(n). Test suffix.
isSuffixOf :: Text -> Text -> Bool

-- | O(n*m). Test infix.
isInfixOf :: Text -> Text -> Bool

-- | O(n). Strip whitespace from both ends.
strip :: Text -> Text

-- | O(n). Strip whitespace from start.
stripStart :: Text -> Text

-- | O(n). Strip whitespace from end.
stripEnd :: Text -> Text

-- | O(n). Strip prefix if present.
stripPrefix :: Text -> Text -> Maybe Text

-- | O(n). Strip suffix if present.
stripSuffix :: Text -> Text -> Maybe Text

-- | O(n). Find common prefixes.
commonPrefixes :: Text -> Text -> Maybe (Text, Text, Text)

-- | O(n). Encode to UTF-8 bytes.
encodeUtf8 :: Text -> Bytes

-- | O(n). Decode from UTF-8 (unsafe on invalid).
decodeUtf8 :: Bytes -> Text

-- | O(n). Decode with error handling.
decodeUtf8' :: Bytes -> Either String Text

-- | O(n). Decode with custom error handler.
decodeUtf8With :: (String -> Maybe Char) -> Bytes -> Text

-- | Read entire file as text.
readFile :: FilePath -> IO Text

-- | Write text to file.
writeFile :: FilePath -> Text -> IO ()

-- | Append text to file.
appendFile :: FilePath -> Text -> IO ()

-- | Interact with stdin/stdout.
interact :: (Text -> Text) -> IO ()

-- | Read line from stdin.
getLine :: IO Text

-- | Read all of stdin.
getContents :: IO Text

-- | Write text to stdout.
putStr :: Text -> IO ()

-- | Write text to stdout with newline.
putStrLn :: Text -> IO ()

-- | Read all contents from handle.
hGetContents :: Handle -> IO Text

-- | Read line from handle.
hGetLine :: Handle -> IO Text

-- | Write text to handle.
hPutStr :: Handle -> Text -> IO ()

-- | Write text to handle with newline.
hPutStrLn :: Handle -> Text -> IO ()

-- This is a specification file.
-- Actual implementation provided by the compiler.
