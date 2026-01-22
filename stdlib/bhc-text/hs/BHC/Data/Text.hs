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

-- Basic interface
cons :: Char -> Text -> Text
cons c t = singleton c `append` t

snoc :: Text -> Char -> Text
snoc t c = t `append` singleton c

foreign import ccall "bhc_text_append" append :: Text -> Text -> Text

uncons :: Text -> Maybe (Char, Text)
uncons t
    | null t    = Nothing
    | otherwise = Just (head t, tail t)

unsnoc :: Text -> Maybe (Text, Char)
unsnoc t
    | null t    = Nothing
    | otherwise = Just (init t, last t)

foreign import ccall "bhc_text_head" head :: Text -> Char
foreign import ccall "bhc_text_last" last :: Text -> Char
foreign import ccall "bhc_text_tail" tail :: Text -> Text
foreign import ccall "bhc_text_init" init :: Text -> Text
foreign import ccall "bhc_text_null" null :: Text -> Bool
foreign import ccall "bhc_text_length" length :: Text -> Int

compareLength :: Text -> Int -> Ordering
compareLength t n = compare (length t) n

-- Transformations
foreign import ccall "bhc_text_map" map :: (Char -> Char) -> Text -> Text

intercalate :: Text -> [Text] -> Text
intercalate sep = concat . go
  where go []     = []
        go [x]    = [x]
        go (x:xs) = x : sep : go xs

intersperse :: Char -> Text -> Text
intersperse c = pack . go . unpack
  where go []     = []
        go [x]    = [x]
        go (x:xs) = x : c : go xs

foreign import ccall "bhc_text_reverse" reverse :: Text -> Text

replace :: Text -> Text -> Text -> Text
replace needle replacement haystack = intercalate replacement (splitOn needle haystack)

foreign import ccall "bhc_text_to_case_fold" toCaseFold :: Text -> Text
foreign import ccall "bhc_text_to_lower" toLower :: Text -> Text
foreign import ccall "bhc_text_to_upper" toUpper :: Text -> Text
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

-- Substrings
foreign import ccall "bhc_text_take" take :: Int -> Text -> Text
foreign import ccall "bhc_text_take_end" takeEnd :: Int -> Text -> Text
foreign import ccall "bhc_text_drop" drop :: Int -> Text -> Text
foreign import ccall "bhc_text_drop_end" dropEnd :: Int -> Text -> Text

takeWhile :: (Char -> Bool) -> Text -> Text
takeWhile p = pack . P.takeWhile p . unpack

takeWhileEnd :: (Char -> Bool) -> Text -> Text
takeWhileEnd p = reverse . takeWhile p . reverse

dropWhile :: (Char -> Bool) -> Text -> Text
dropWhile p = pack . P.dropWhile p . unpack

dropWhileEnd :: (Char -> Bool) -> Text -> Text
dropWhileEnd p = reverse . dropWhile p . reverse

dropAround :: (Char -> Bool) -> Text -> Text
dropAround p = dropWhile p . dropWhileEnd p

strip :: Text -> Text
strip = dropAround isSpace
  where isSpace c = c `P.elem` " \t\n\r"

stripStart :: Text -> Text
stripStart = dropWhile isSpace
  where isSpace c = c `P.elem` " \t\n\r"

stripEnd :: Text -> Text
stripEnd = dropWhileEnd isSpace
  where isSpace c = c `P.elem` " \t\n\r"

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

-- Predicates
foreign import ccall "bhc_text_is_prefix_of" isPrefixOf :: Text -> Text -> Bool
foreign import ccall "bhc_text_is_suffix_of" isSuffixOf :: Text -> Text -> Bool
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
