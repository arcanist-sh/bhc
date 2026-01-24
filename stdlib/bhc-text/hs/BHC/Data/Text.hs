-- |
-- Module      : BHC.Data.Text
-- Description : Efficient packed Unicode text
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- An efficient packed representation of Unicode text. This module
-- provides SIMD-accelerated operations where available.

module BHC.Data.Text (
    -- * Types
    Text,
    
    -- * Creation and elimination
    pack, unpack, singleton, empty,
    
    -- * Basic interface
    cons, snoc, append, uncons, unsnoc,
    head, last, tail, init, null, length, compareLength,
    
    -- * Transformations
    map, intercalate, intersperse,
    reverse, replace,
    toCaseFold, toLower, toUpper, toTitle,
    justifyLeft, justifyRight, center,
    
    -- * Folds
    foldl, foldl', foldl1, foldl1',
    foldr, foldr1,
    
    -- * Special folds
    concat, concatMap, any, all, maximum, minimum,
    
    -- * Substrings
    take, takeEnd, drop, dropEnd,
    takeWhile, takeWhileEnd, dropWhile, dropWhileEnd, dropAround,
    strip, stripStart, stripEnd,
    splitAt, breakOn, breakOnEnd, break, span,
    
    -- * Breaking into many
    splitOn, split, chunksOf,
    lines, words, unlines, unwords,
    
    -- * Predicates
    isPrefixOf, isSuffixOf, isInfixOf,
    
    -- * View patterns
    stripPrefix, stripSuffix, commonPrefixes,
    
    -- * Searching
    filter, breakOnAll, find, elem, partition,
    
    -- * Indexing
    index, findIndex, count,
    
    -- * Zipping
    zip, zipWith,
    
    -- * Low level
    copy,
) where

import BHC.Prelude hiding (
    head, last, tail, init, null, length, map, reverse,
    foldl, foldl1, foldr, foldr1, concat, concatMap, any, all,
    maximum, minimum, take, drop, takeWhile, dropWhile, break, span,
    splitAt, lines, words, unlines, unwords, filter, elem, zip, zipWith
    )
import qualified BHC.Prelude as P

-- | A space efficient, packed, unboxed Unicode text type.
data Text = Text
    {-# UNPACK #-} !ByteArray  -- UTF-8 bytes
    {-# UNPACK #-} !Int        -- Offset
    {-# UNPACK #-} !Int        -- Length

instance Eq Text where
    (==) = eqText

instance Ord Text where
    compare = compareText

instance Show Text where
    showsPrec p t = showsPrec p (unpack t)

instance Read Text where
    readsPrec p s = [(pack t, r) | (t, r) <- readsPrec p s]

instance Semigroup Text where
    (<>) = append

instance Monoid Text where
    mempty = empty

instance P.IsString Text where
    fromString = pack

-- Primitives (FFI to Rust)
foreign import ccall "bhc_text_pack" pack :: String -> Text
foreign import ccall "bhc_text_unpack" unpack :: Text -> String
foreign import ccall "bhc_text_empty" empty :: Text
foreign import ccall "bhc_text_singleton" singleton :: Char -> Text
foreign import ccall "bhc_text_eq" eqText :: Text -> Text -> Bool
foreign import ccall "bhc_text_compare" compareText :: Text -> Text -> Ordering

-- ------------------------------------------------------------
-- Basic interface
-- ------------------------------------------------------------

-- | /O(n)/. Prepend a character to a 'Text'.
--
-- >>> cons 'H' (pack "ello")
-- "Hello"
cons :: Char -> Text -> Text
cons c t = singleton c `append` t

-- | /O(n)/. Append a character to a 'Text'.
--
-- >>> snoc (pack "Hell") 'o'
-- "Hello"
snoc :: Text -> Char -> Text
snoc t c = t `append` singleton c

-- | /O(n)/. Append two 'Text' values.
--
-- >>> append (pack "Hello") (pack " World")
-- "Hello World"
foreign import ccall "bhc_text_append" append :: Text -> Text -> Text

-- | /O(1)/. Decompose a 'Text' into its first character and the rest.
-- Returns 'Nothing' if the 'Text' is empty.
--
-- >>> uncons (pack "Hello")
-- Just ('H',"ello")
-- >>> uncons empty
-- Nothing
uncons :: Text -> Maybe (Char, Text)
uncons t
    | null t    = Nothing
    | otherwise = Just (head t, tail t)

-- | /O(1)/. Decompose a 'Text' into its initial portion and last character.
-- Returns 'Nothing' if the 'Text' is empty.
--
-- >>> unsnoc (pack "Hello")
-- Just ("Hell",'o')
unsnoc :: Text -> Maybe (Text, Char)
unsnoc t
    | null t    = Nothing
    | otherwise = Just (init t, last t)

-- | /O(1)/. Extract the first character of a 'Text'.
--
-- __Warning__: Partial function. Throws an error on empty 'Text'.
--
-- >>> head (pack "Hello")
-- 'H'
foreign import ccall "bhc_text_head" head :: Text -> Char

-- | /O(1)/. Extract the last character of a 'Text'.
--
-- __Warning__: Partial function. Throws an error on empty 'Text'.
--
-- >>> last (pack "Hello")
-- 'o'
foreign import ccall "bhc_text_last" last :: Text -> Char

-- | /O(1)/. Return all characters after the head of a 'Text'.
--
-- __Warning__: Partial function. Throws an error on empty 'Text'.
--
-- >>> tail (pack "Hello")
-- "ello"
foreign import ccall "bhc_text_tail" tail :: Text -> Text

-- | /O(1)/. Return all characters except the last of a 'Text'.
--
-- __Warning__: Partial function. Throws an error on empty 'Text'.
--
-- >>> init (pack "Hello")
-- "Hell"
foreign import ccall "bhc_text_init" init :: Text -> Text

-- | /O(1)/. Test whether a 'Text' is empty.
--
-- >>> null empty
-- True
-- >>> null (pack "Hello")
-- False
foreign import ccall "bhc_text_null" null :: Text -> Bool

-- | /O(1)/. Return the length of a 'Text' in characters.
--
-- >>> length (pack "Hello")
-- 5
foreign import ccall "bhc_text_length" length :: Text -> Int

-- | /O(1)/. Compare the length of a 'Text' to an 'Int'.
-- More efficient than @compare (length t) n@ when you only care
-- about the ordering.
--
-- >>> compareLength (pack "Hello") 3
-- GT
compareLength :: Text -> Int -> Ordering
compareLength t n = compare (length t) n

-- ------------------------------------------------------------
-- Transformations
-- ------------------------------------------------------------

-- | /O(n)/. Apply a function to each character in a 'Text'.
--
-- >>> map toUpper (pack "hello")
-- "HELLO"
foreign import ccall "bhc_text_map" map :: (Char -> Char) -> Text -> Text

-- | /O(n)/. Join a list of 'Text' values with a separator.
--
-- >>> intercalate (pack ", ") [pack "one", pack "two", pack "three"]
-- "one, two, three"
intercalate :: Text -> [Text] -> Text
intercalate sep = concat . go
  where go []     = []
        go [x]    = [x]
        go (x:xs) = x : sep : go xs

-- | /O(n)/. Insert a character between adjacent characters.
--
-- >>> intersperse '-' (pack "HELLO")
-- "H-E-L-L-O"
intersperse :: Char -> Text -> Text
intersperse c = pack . go . unpack
  where go []     = []
        go [x]    = [x]
        go (x:xs) = x : c : go xs

-- | /O(n)/. Reverse a 'Text'.
--
-- >>> reverse (pack "Hello")
-- "olleH"
foreign import ccall "bhc_text_reverse" reverse :: Text -> Text

-- | /O(n*m)/. Replace all occurrences of a needle with a replacement.
--
-- >>> replace (pack "world") (pack "Haskell") (pack "Hello world")
-- "Hello Haskell"
replace :: Text -> Text -> Text -> Text
replace needle replacement haystack = intercalate replacement (splitOn needle haystack)

-- | /O(n)/. Case fold: convert to a canonical lowercase form for
-- case-insensitive comparisons.
--
-- >>> toCaseFold (pack "Hello WORLD")
-- "hello world"
foreign import ccall "bhc_text_to_case_fold" toCaseFold :: Text -> Text

-- | /O(n)/. Convert to lowercase.
--
-- >>> toLower (pack "Hello WORLD")
-- "hello world"
foreign import ccall "bhc_text_to_lower" toLower :: Text -> Text

-- | /O(n)/. Convert to uppercase.
--
-- >>> toUpper (pack "Hello World")
-- "HELLO WORLD"
foreign import ccall "bhc_text_to_upper" toUpper :: Text -> Text

-- | /O(n)/. Convert to title case (first letter of each word uppercase).
--
-- >>> toTitle (pack "hello world")
-- "Hello World"
foreign import ccall "bhc_text_to_title" toTitle :: Text -> Text

justifyLeft :: Int -> Char -> Text -> Text
justifyLeft n c t
    | len >= n  = t
    | otherwise = t `append` pack (P.replicate (n - len) c)
  where len = length t

justifyRight :: Int -> Char -> Text -> Text
justifyRight n c t
    | len >= n  = t
    | otherwise = pack (P.replicate (n - len) c) `append` t
  where len = length t

center :: Int -> Char -> Text -> Text
center n c t
    | len >= n  = t
    | otherwise = pack (P.replicate left c) `append` t `append` pack (P.replicate right c)
  where len = length t
        pad = n - len
        left = pad `div` 2
        right = pad - left

-- Folds
foldl :: (a -> Char -> a) -> a -> Text -> a
foldl f z = P.foldl f z . unpack

foldl' :: (a -> Char -> a) -> a -> Text -> a
foldl' f z = P.foldl' f z . unpack

foldl1 :: (Char -> Char -> Char) -> Text -> Char
foldl1 f = P.foldl1 f . unpack

foldl1' :: (Char -> Char -> Char) -> Text -> Char
foldl1' f = P.foldl1 f . unpack

foldr :: (Char -> a -> a) -> a -> Text -> a
foldr f z = P.foldr f z . unpack

foldr1 :: (Char -> Char -> Char) -> Text -> Char
foldr1 f = P.foldr1 f . unpack

-- Special folds
concat :: [Text] -> Text
concat = P.foldl' append empty

concatMap :: (Char -> Text) -> Text -> Text
concatMap f = concat . P.map f . unpack

any :: (Char -> Bool) -> Text -> Bool
any p = P.any p . unpack

all :: (Char -> Bool) -> Text -> Bool
all p = P.all p . unpack

maximum :: Text -> Char
maximum = P.maximum . unpack

minimum :: Text -> Char
minimum = P.minimum . unpack

-- ------------------------------------------------------------
-- Substrings
-- ------------------------------------------------------------

-- | /O(1)/. Take the first @n@ characters of a 'Text'.
--
-- >>> take 5 (pack "Hello World")
-- "Hello"
foreign import ccall "bhc_text_take" take :: Int -> Text -> Text

-- | /O(1)/. Take the last @n@ characters of a 'Text'.
--
-- >>> takeEnd 5 (pack "Hello World")
-- "World"
foreign import ccall "bhc_text_take_end" takeEnd :: Int -> Text -> Text

-- | /O(1)/. Drop the first @n@ characters of a 'Text'.
--
-- >>> drop 6 (pack "Hello World")
-- "World"
foreign import ccall "bhc_text_drop" drop :: Int -> Text -> Text

-- | /O(1)/. Drop the last @n@ characters of a 'Text'.
--
-- >>> dropEnd 6 (pack "Hello World")
-- "Hello"
foreign import ccall "bhc_text_drop_end" dropEnd :: Int -> Text -> Text

-- | /O(n)/. Take characters while the predicate holds.
--
-- >>> takeWhile (/= ' ') (pack "Hello World")
-- "Hello"
takeWhile :: (Char -> Bool) -> Text -> Text
takeWhile p = pack . P.takeWhile p . unpack

-- | /O(n)/. Take characters from the end while the predicate holds.
--
-- >>> takeWhileEnd (/= ' ') (pack "Hello World")
-- "World"
takeWhileEnd :: (Char -> Bool) -> Text -> Text
takeWhileEnd p = reverse . takeWhile p . reverse

-- | /O(n)/. Drop characters while the predicate holds.
--
-- >>> dropWhile (== ' ') (pack "   Hello")
-- "Hello"
dropWhile :: (Char -> Bool) -> Text -> Text
dropWhile p = pack . P.dropWhile p . unpack

-- | /O(n)/. Drop characters from the end while the predicate holds.
--
-- >>> dropWhileEnd (== ' ') (pack "Hello   ")
-- "Hello"
dropWhileEnd :: (Char -> Bool) -> Text -> Text
dropWhileEnd p = reverse . dropWhile p . reverse

-- | /O(n)/. Drop characters from both ends while the predicate holds.
--
-- >>> dropAround (== ' ') (pack "  Hello  ")
-- "Hello"
dropAround :: (Char -> Bool) -> Text -> Text
dropAround p = dropWhile p . dropWhileEnd p

-- | /O(n)/. Remove leading and trailing whitespace.
--
-- >>> strip (pack "  Hello World  ")
-- "Hello World"
strip :: Text -> Text
strip = dropAround isSpace
  where isSpace c = c `P.elem` " \t\n\r"

-- | /O(n)/. Remove leading whitespace.
--
-- >>> stripStart (pack "  Hello")
-- "Hello"
stripStart :: Text -> Text
stripStart = dropWhile isSpace
  where isSpace c = c `P.elem` " \t\n\r"

-- | /O(n)/. Remove trailing whitespace.
--
-- >>> stripEnd (pack "Hello  ")
-- "Hello"
stripEnd :: Text -> Text
stripEnd = dropWhileEnd isSpace
  where isSpace c = c `P.elem` " \t\n\r"

-- | /O(1)/. Split a 'Text' at the given position.
--
-- >>> splitAt 5 (pack "Hello World")
-- ("Hello"," World")
splitAt :: Int -> Text -> (Text, Text)
splitAt n t = (take n t, drop n t)

breakOn :: Text -> Text -> (Text, Text)
breakOn needle haystack = case indexOf needle haystack of
    Nothing -> (haystack, empty)
    Just i  -> (take i haystack, drop i haystack)

breakOnEnd :: Text -> Text -> (Text, Text)
breakOnEnd needle haystack = case lastIndexOf needle haystack of
    Nothing -> (empty, haystack)
    Just i  -> (take (i + length needle) haystack, drop (i + length needle) haystack)

break :: (Char -> Bool) -> Text -> (Text, Text)
break p t = span (not . p) t

span :: (Char -> Bool) -> Text -> (Text, Text)
span p t = (takeWhile p t, dropWhile p t)

-- Breaking into many
splitOn :: Text -> Text -> [Text]
splitOn needle haystack
    | null needle = error "Text.splitOn: empty delimiter"
    | null haystack = [empty]
    | otherwise = go haystack
  where
    go t = case breakOn needle t of
        (before, after)
            | null after -> [before]
            | otherwise  -> before : go (drop (length needle) after)

split :: (Char -> Bool) -> Text -> [Text]
split p = P.map pack . go . unpack
  where
    go []  = [[]]
    go (c:cs)
        | p c       = [] : go cs
        | otherwise = let (w:ws) = go cs in (c:w) : ws

chunksOf :: Int -> Text -> [Text]
chunksOf k = go
  where
    go t
        | null t    = []
        | otherwise = take k t : go (drop k t)

lines :: Text -> [Text]
lines = split (== '\n')

words :: Text -> [Text]
words = P.filter (not . null) . split isSpace
  where isSpace c = c `P.elem` " \t\n\r"

unlines :: [Text] -> Text
unlines = concat . P.map (`snoc` '\n')

unwords :: [Text] -> Text
unwords = intercalate (singleton ' ')

-- ------------------------------------------------------------
-- Predicates
-- ------------------------------------------------------------

-- | /O(n)/. Test whether the first 'Text' is a prefix of the second.
--
-- >>> isPrefixOf (pack "Hello") (pack "Hello World")
-- True
-- >>> isPrefixOf (pack "World") (pack "Hello World")
-- False
foreign import ccall "bhc_text_is_prefix_of" isPrefixOf :: Text -> Text -> Bool

-- | /O(n)/. Test whether the first 'Text' is a suffix of the second.
--
-- >>> isSuffixOf (pack "World") (pack "Hello World")
-- True
-- >>> isSuffixOf (pack "Hello") (pack "Hello World")
-- False
foreign import ccall "bhc_text_is_suffix_of" isSuffixOf :: Text -> Text -> Bool

-- | /O(n*m)/. Test whether the first 'Text' is contained within the second.
--
-- >>> isInfixOf (pack "lo Wo") (pack "Hello World")
-- True
-- >>> isInfixOf (pack "xyz") (pack "Hello World")
-- False
foreign import ccall "bhc_text_is_infix_of" isInfixOf :: Text -> Text -> Bool

-- View patterns
stripPrefix :: Text -> Text -> Maybe Text
stripPrefix prefix t
    | isPrefixOf prefix t = Just (drop (length prefix) t)
    | otherwise           = Nothing

stripSuffix :: Text -> Text -> Maybe Text
stripSuffix suffix t
    | isSuffixOf suffix t = Just (dropEnd (length suffix) t)
    | otherwise           = Nothing

commonPrefixes :: Text -> Text -> Maybe (Text, Text, Text)
commonPrefixes t1 t2 = go 0
  where
    go i
        | i >= length t1 || i >= length t2 = done i
        | index t1 i /= index t2 i = done i
        | otherwise = go (i + 1)
    done 0 = Nothing
    done i = Just (take i t1, drop i t1, drop i t2)

-- Searching
filter :: (Char -> Bool) -> Text -> Text
filter p = pack . P.filter p . unpack

breakOnAll :: Text -> Text -> [(Text, Text)]
breakOnAll needle haystack = go 0 haystack
  where
    go _ t | null t = []
    go offset t = case indexOf needle t of
        Nothing -> []
        Just i  -> 
            let before = take (offset + i) haystack
                after = drop (offset + i) haystack
            in (before, after) : go (offset + i + length needle) (drop (i + length needle) t)

find :: (Char -> Bool) -> Text -> Maybe Char
find p = P.find p . unpack

elem :: Char -> Text -> Bool
elem c = P.elem c . unpack

partition :: (Char -> Bool) -> Text -> (Text, Text)
partition p t = (filter p t, filter (not . p) t)

-- Indexing
index :: Text -> Int -> Char
index t i
    | i < 0 || i >= length t = error "Text.index: index out of bounds"
    | otherwise = unpack t !! i

findIndex :: (Char -> Bool) -> Text -> Maybe Int
findIndex p = P.findIndex p . unpack

count :: Text -> Text -> Int
count needle = P.length . breakOnAll needle

-- Zipping
zip :: Text -> Text -> [(Char, Char)]
zip t1 t2 = P.zip (unpack t1) (unpack t2)

zipWith :: (Char -> Char -> Char) -> Text -> Text -> Text
zipWith f t1 t2 = pack (P.zipWith f (unpack t1) (unpack t2))

-- Low level
copy :: Text -> Text
copy t = pack (unpack t)

-- Internal helpers
indexOf :: Text -> Text -> Maybe Int
indexOf needle haystack = findIndex (\i -> isPrefixOf needle (drop i haystack)) [0..length haystack - length needle]
  where findIndex p [] = Nothing
        findIndex p (x:xs) | p x = Just x | otherwise = findIndex p xs

lastIndexOf :: Text -> Text -> Maybe Int
lastIndexOf needle haystack = go Nothing 0
  where
    go result offset
        | offset > length haystack - length needle = result
        | isPrefixOf needle (drop offset haystack) = go (Just offset) (offset + 1)
        | otherwise = go result (offset + 1)

-- Opaque type for FFI
data ByteArray
