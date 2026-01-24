-- |
-- Module      : H26.Bytes
-- Description : Byte arrays with slicing and pinned memory support
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Efficient byte array operations for binary data processing.
--
-- = Overview
--
-- 'Bytes' is an immutable byte array type with O(1) slicing.
-- Supports both managed and pinned memory for FFI interop.
--
-- = Quick Start
--
-- @
-- import H26.Bytes
--
-- -- Construction
-- bs = pack [0x48, 0x65, 0x6c, 0x6c, 0x6f]  -- \"Hello\" in ASCII
-- bs' = replicate 100 0                      -- 100 zero bytes
--
-- -- Slicing (O(1), creates view)
-- header = take 4 fileContents
-- payload = drop 4 fileContents
--
-- -- File I\/O
-- contents <- readFile \"binary.dat\"
-- writeFile \"output.dat\" processedData
-- @
--
-- = Memory Types
--
-- The module provides three byte array types:
--
-- [@Bytes@] Immutable, GC-managed byte array
-- [@MutableBytes@] Mutable byte array for in-place operations
-- [@PinnedBytes@] Memory that won't move (required for FFI)
--
-- = FFI Interop
--
-- For calling C code, use 'PinnedBytes' to ensure pointers remain valid:
--
-- @
-- import H26.Bytes
-- import H26.FFI
--
-- processWithC :: Bytes -> IO Bytes
-- processWithC bs = do
--     let pinned = toPinned bs
--     withPinnedPtr pinned $ \\ptr -> do
--         c_process ptr (length bs)
--         -- ptr is valid here
--     return (fromPinned pinned)
-- @
--
-- = See Also
--
-- * "H26.Text" for Unicode text (uses UTF-8 bytes internally)
-- * "H26.FFI" for foreign function interface
-- * "BHC.Data.ByteString" for the underlying implementation

{-# HASKELL_EDITION 2026 #-}

module H26.Bytes
  ( -- * Byte Array Types
    Bytes
  , MutableBytes
  , PinnedBytes

    -- * Construction
  , empty
  , singleton
  , pack
  , unpack
  , replicate
  , generate

    -- * Basic Operations
  , length
  , null
  , index
  , (!?)
  , head
  , tail
  , last
  , init

    -- * Slicing (Views)
  , take
  , drop
  , splitAt
  , slice
  , takeWhile
  , dropWhile
  , span
  , break

    -- * Combining
  , append
  , concat
  , intercalate

    -- * Searching
  , elem
  , notElem
  , find
  , findIndex
  , findIndices
  , elemIndex
  , elemIndices

    -- * Transformations
  , map
  , reverse
  , intersperse
  , transpose

    -- * Folds
  , foldl
  , foldl'
  , foldr
  , foldl1
  , foldr1

    -- * Special Folds
  , all
  , any
  , maximum
  , minimum
  , sum

    -- * Building
  , scanl
  , scanr
  , unfoldr

    -- * Zipping
  , zip
  , zipWith
  , unzip

    -- * Pinned Memory (FFI Support)
  , toPinned
  , fromPinned
  , withPinnedPtr
  , unsafePinnedPtr

    -- * Mutable Operations
  , new
  , read
  , write
  , modify
  , freeze
  , thaw
  , copy

    -- * Conversions
  , toList
  , fromList
  , toStrict
  , fromStrict

    -- * Encoding/Decoding
  , encodeUtf8
  , decodeUtf8
  , decodeUtf8'
  , encodeBase64
  , decodeBase64

    -- * I/O
  , readFile
  , writeFile
  , appendFile
  , hGet
  , hPut
  , hGetLine
  , hGetContents
  ) where

import Prelude hiding
  ( length, null, head, tail, last, init
  , take, drop, splitAt, takeWhile, dropWhile, span, break
  , elem, notElem, map, reverse, foldl, foldr, all, any
  , maximum, minimum, sum, concat, zip, zipWith, unzip
  , replicate, readFile, writeFile, appendFile, read
  )

-- | Immutable byte array.
--
-- Compact representation of binary data. Slicing creates views
-- without copying.
data Bytes

-- | Mutable byte array in IO or ST.
data MutableBytes s

-- | Pinned byte array that will not be moved by GC.
--
-- Required for FFI interop where C code holds pointers.
data PinnedBytes

-- | O(1). The empty byte array.
empty :: Bytes

-- | O(1). A byte array with a single element.
singleton :: Word8 -> Bytes

-- | O(n). Pack a list of bytes.
pack :: [Word8] -> Bytes

-- | O(n). Unpack to a list.
unpack :: Bytes -> [Word8]

-- | O(n). Replicate a byte n times.
replicate :: Int -> Word8 -> Bytes

-- | O(n). Generate bytes using a function.
generate :: Int -> (Int -> Word8) -> Bytes

-- | O(1). Length in bytes.
length :: Bytes -> Int

-- | O(1). Test if empty.
null :: Bytes -> Bool

-- | O(1). Index a byte (unsafe).
index :: Bytes -> Int -> Word8

-- | O(1). Safe indexing.
(!?) :: Bytes -> Int -> Maybe Word8

-- | O(1). First byte (unsafe on empty).
head :: Bytes -> Word8

-- | O(1). All but the first byte (view).
tail :: Bytes -> Bytes

-- | O(1). Last byte (unsafe on empty).
last :: Bytes -> Word8

-- | O(1). All but the last byte (view).
init :: Bytes -> Bytes

-- | O(1). Take first n bytes (view).
take :: Int -> Bytes -> Bytes

-- | O(1). Drop first n bytes (view).
drop :: Int -> Bytes -> Bytes

-- | O(1). Split at position (views).
splitAt :: Int -> Bytes -> (Bytes, Bytes)

-- | O(1). Slice from start to end indices (view).
slice :: Int -> Int -> Bytes -> Bytes

-- | O(n). Take while predicate holds.
takeWhile :: (Word8 -> Bool) -> Bytes -> Bytes

-- | O(n). Drop while predicate holds.
dropWhile :: (Word8 -> Bool) -> Bytes -> Bytes

-- | O(n). Split at first element not satisfying predicate.
span :: (Word8 -> Bool) -> Bytes -> (Bytes, Bytes)

-- | O(n). Split at first element satisfying predicate.
break :: (Word8 -> Bool) -> Bytes -> (Bytes, Bytes)

-- | O(n). Append two byte arrays.
append :: Bytes -> Bytes -> Bytes

-- | O(n). Concatenate a list of byte arrays.
concat :: [Bytes] -> Bytes

-- | O(n). Intercalate a separator.
intercalate :: Bytes -> [Bytes] -> Bytes

-- | O(n). Test membership.
elem :: Word8 -> Bytes -> Bool

-- | O(n). Test non-membership.
notElem :: Word8 -> Bytes -> Bool

-- | O(n). Find first element satisfying predicate.
find :: (Word8 -> Bool) -> Bytes -> Maybe Word8

-- | O(n). Find index of first element satisfying predicate.
findIndex :: (Word8 -> Bool) -> Bytes -> Maybe Int

-- | O(n). Find all indices of elements satisfying predicate.
findIndices :: (Word8 -> Bool) -> Bytes -> [Int]

-- | O(n). Find index of element.
elemIndex :: Word8 -> Bytes -> Maybe Int

-- | O(n). Find all indices of element.
elemIndices :: Word8 -> Bytes -> [Int]

-- | O(n). Map a function over bytes.
map :: (Word8 -> Word8) -> Bytes -> Bytes

-- | O(n). Reverse byte order.
reverse :: Bytes -> Bytes

-- | O(n). Intersperse a byte between elements.
intersperse :: Word8 -> Bytes -> Bytes

-- | O(n*m). Transpose rows and columns.
transpose :: [Bytes] -> [Bytes]

-- | O(n). Left fold.
foldl :: (a -> Word8 -> a) -> a -> Bytes -> a

-- | O(n). Strict left fold.
foldl' :: (a -> Word8 -> a) -> a -> Bytes -> a

-- | O(n). Right fold.
foldr :: (Word8 -> a -> a) -> a -> Bytes -> a

-- | O(n). Left fold without starting value.
foldl1 :: (Word8 -> Word8 -> Word8) -> Bytes -> Word8

-- | O(n). Right fold without starting value.
foldr1 :: (Word8 -> Word8 -> Word8) -> Bytes -> Word8

-- | O(n). Test if all elements satisfy predicate.
all :: (Word8 -> Bool) -> Bytes -> Bool

-- | O(n). Test if any element satisfies predicate.
any :: (Word8 -> Bool) -> Bytes -> Bool

-- | O(n). Maximum element.
maximum :: Bytes -> Word8

-- | O(n). Minimum element.
minimum :: Bytes -> Word8

-- | O(n). Sum of elements.
sum :: Bytes -> Int

-- | O(n). Scan left.
scanl :: (Word8 -> Word8 -> Word8) -> Word8 -> Bytes -> Bytes

-- | O(n). Scan right.
scanr :: (Word8 -> Word8 -> Word8) -> Word8 -> Bytes -> Bytes

-- | O(n). Build from unfolding function.
unfoldr :: (a -> Maybe (Word8, a)) -> a -> Bytes

-- | O(n). Zip two byte arrays.
zip :: Bytes -> Bytes -> [(Word8, Word8)]

-- | O(n). Zip with function.
zipWith :: (Word8 -> Word8 -> a) -> Bytes -> Bytes -> [a]

-- | O(n). Unzip pairs.
unzip :: [(Word8, Word8)] -> (Bytes, Bytes)

-- | O(n). Copy to pinned memory.
--
-- The resulting bytes will not be moved by GC, suitable for FFI.
toPinned :: Bytes -> PinnedBytes

-- | O(1). Convert pinned to regular (view).
fromPinned :: PinnedBytes -> Bytes

-- | Execute action with pointer to pinned data.
--
-- The pointer is valid only for the duration of the action.
withPinnedPtr :: PinnedBytes -> (Ptr Word8 -> IO a) -> IO a

-- | Get raw pointer (unsafe, caller manages lifetime).
unsafePinnedPtr :: PinnedBytes -> Ptr Word8

-- | O(n). Allocate new mutable byte array.
new :: Int -> IO (MutableBytes RealWorld)

-- | O(1). Read byte at index.
read :: MutableBytes s -> Int -> ST s Word8

-- | O(1). Write byte at index.
write :: MutableBytes s -> Int -> Word8 -> ST s ()

-- | O(1). Modify byte at index.
modify :: MutableBytes s -> Int -> (Word8 -> Word8) -> ST s ()

-- | O(n). Freeze mutable to immutable.
freeze :: MutableBytes s -> ST s Bytes

-- | O(n). Thaw immutable to mutable.
thaw :: Bytes -> ST s (MutableBytes s)

-- | O(n). Copy bytes between mutable arrays.
copy :: MutableBytes s -> Int -> MutableBytes s -> Int -> Int -> ST s ()

-- | O(n). Convert to list.
toList :: Bytes -> [Word8]

-- | O(n). Convert from list.
fromList :: [Word8] -> Bytes

-- | O(1). Convert lazy to strict.
toStrict :: Bytes -> Bytes

-- | O(1). Convert strict to lazy.
fromStrict :: Bytes -> Bytes

-- | Encode Text to UTF-8 bytes.
encodeUtf8 :: Text -> Bytes

-- | Decode UTF-8 bytes to Text (unsafe on invalid).
decodeUtf8 :: Bytes -> Text

-- | Decode UTF-8 with error handling.
decodeUtf8' :: Bytes -> Either String Text

-- | Encode to Base64.
encodeBase64 :: Bytes -> Bytes

-- | Decode from Base64.
decodeBase64 :: Bytes -> Either String Bytes

-- | Read entire file as bytes.
readFile :: FilePath -> IO Bytes

-- | Write bytes to file.
writeFile :: FilePath -> Bytes -> IO ()

-- | Append bytes to file.
appendFile :: FilePath -> Bytes -> IO ()

-- | Read n bytes from handle.
hGet :: Handle -> Int -> IO Bytes

-- | Write bytes to handle.
hPut :: Handle -> Bytes -> IO ()

-- | Read line from handle.
hGetLine :: Handle -> IO Bytes

-- | Read all remaining contents.
hGetContents :: Handle -> IO Bytes

-- This is a specification file.
-- Actual implementation provided by the compiler.
