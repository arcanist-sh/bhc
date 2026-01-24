-- |
-- Module      : H26.Vector
-- Description : Boxed and unboxed vector types
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Efficient array-like containers with O(1) indexing.
--
-- = Overview
--
-- This module provides two vector types:
--
-- * 'Vector' — Boxed vectors that can hold any type
-- * 'UVector' — Unboxed vectors for primitive types (better performance)
--
-- Both support O(1) indexing and O(1) slicing (creating views).
--
-- = Quick Start
--
-- @
-- import H26.Vector
--
-- -- Construction
-- v1 = fromList [1, 2, 3, 4, 5]
-- v2 = replicate 100 0
-- v3 = generate 100 (\\i -> i * i)
--
-- -- Indexing
-- x = v1 ! 2               -- 3 (unsafe)
-- y = v1 !? 10             -- Nothing (safe)
--
-- -- Slicing (O(1), creates view)
-- prefix = take 3 v1       -- [1, 2, 3]
-- suffix = drop 2 v1       -- [3, 4, 5]
--
-- -- Transformations
-- doubled = map (*2) v1    -- [2, 4, 6, 8, 10]
-- evens = filter even v1   -- [2, 4]
--
-- -- Folds
-- total = sum v1           -- 15
-- big = maximum v1         -- 5
-- @
--
-- = Boxed vs Unboxed
--
-- Choose 'UVector' for primitive types when performance matters:
--
-- @
-- -- Boxed: each element is a pointer (8 bytes overhead)
-- boxed :: Vector Int
-- boxed = fromList [1..1000000]
--
-- -- Unboxed: elements stored directly (no overhead)
-- unboxed :: UVector Int
-- unboxed = fromList [1..1000000]  -- ~8x less memory
-- @
--
-- 'UVector' requires an 'Unbox' instance for the element type.
--
-- = Mutable Vectors
--
-- For in-place modifications, use 'MVector' or 'MUVector':
--
-- @
-- import Control.Monad.ST
--
-- sortInPlace :: UVector Int -> UVector Int
-- sortInPlace v = runST $ do
--     mv <- thaw v
--     quicksort mv
--     freeze mv
-- @
--
-- = See Also
--
-- * "H26.Tensor" for multidimensional arrays
-- * "H26.Bytes" for raw byte arrays
-- * "BHC.Data.Vector" for the underlying implementation

{-# HASKELL_EDITION 2026 #-}

module H26.Vector
  ( -- * Boxed Vectors
    Vector

    -- * Unboxed Vectors
  , UVector
  , Unbox(..)

    -- * Construction
  , empty
  , singleton
  , replicate
  , generate
  , iterateN
  , replicateM
  , generateM
  , fromList
  , fromListN

    -- * Basic Operations
  , length
  , null
  , (!)
  , (!?)
  , head
  , last
  , unsafeIndex
  , unsafeHead
  , unsafeLast

    -- * Slicing (Views)
  , slice
  , init
  , tail
  , take
  , drop
  , splitAt

    -- * Construction from Vectors
  , cons
  , snoc
  , (++)
  , concat
  , force

    -- * Transformations
  , map
  , imap
  , concatMap
  , mapM
  , mapM_
  , forM
  , forM_

    -- * Zipping
  , zipWith
  , zipWith3
  , zipWith4
  , zipWithM
  , zipWithM_
  , zip
  , zip3
  , zip4
  , unzip
  , unzip3
  , unzip4

    -- * Filtering
  , filter
  , ifilter
  , filterM
  , takeWhile
  , dropWhile
  , partition
  , span
  , break

    -- * Searching
  , elem
  , notElem
  , find
  , findIndex
  , findIndices
  , elemIndex
  , elemIndices

    -- * Folding
  , foldl
  , foldl'
  , foldl1
  , foldl1'
  , foldr
  , foldr'
  , foldr1
  , foldr1'
  , ifoldl
  , ifoldl'
  , ifoldr
  , ifoldr'

    -- * Specialized Folds
  , all
  , any
  , sum
  , product
  , maximum
  , minimum
  , maximumBy
  , minimumBy
  , maxIndex
  , minIndex

    -- * Scans
  , prescanl
  , prescanl'
  , postscanl
  , postscanl'
  , scanl
  , scanl'
  , scanl1
  , scanl1'
  , prescanr
  , prescanr'
  , postscanr
  , postscanr'
  , scanr
  , scanr'
  , scanr1
  , scanr1'

    -- * Conversion
  , toList
  , convert

    -- * Mutable Vectors
  , MVector
  , MUVector
  , new
  , newWith
  , read
  , write
  , modify
  , swap
  , grow
  , freeze
  , thaw
  , copy
  , unsafeFreeze
  , unsafeThaw

    -- * Sorting
  , sort
  , sortBy
  , sortOn

    -- * Unboxed Type Class
  , Unbox
  ) where

import Prelude hiding
  ( length, null, head, last, init, tail, take, drop, splitAt
  , (++), concat, map, concatMap, mapM, mapM_
  , zipWith, zipWith3, zip, zip3, unzip, unzip3
  , filter, takeWhile, dropWhile, span, break
  , elem, notElem, foldl, foldr, all, any, sum, product
  , maximum, minimum, scanl, scanl1, scanr, scanr1
  , replicate, read
  )

-- | Boxed vector - can contain any type.
data Vector a

-- | Unboxed vector - elements stored without indirection.
--
-- More efficient for primitive types (Int, Float, etc.)
-- but element type must be an instance of Unbox.
data UVector a

-- | Mutable boxed vector.
data MVector s a

-- | Mutable unboxed vector.
data MUVector s a

-- | Class for types that can be stored in unboxed vectors.
class Unbox a where
  -- | Size of element in bytes.
  sizeOf :: proxy a -> Int
  -- | Alignment requirement.
  alignment :: proxy a -> Int

-- | O(1). Empty vector.
empty :: Vector a

-- | O(1). Single element vector.
singleton :: a -> Vector a

-- | O(n). Replicate element n times.
replicate :: Int -> a -> Vector a

-- | O(n). Generate using index function.
generate :: Int -> (Int -> a) -> Vector a

-- | O(n). Iterate function n times.
iterateN :: Int -> (a -> a) -> a -> Vector a

-- | O(n). Monadic replicate.
replicateM :: Monad m => Int -> m a -> m (Vector a)

-- | O(n). Monadic generate.
generateM :: Monad m => Int -> (Int -> m a) -> m (Vector a)

-- | O(n). Convert list to vector.
fromList :: [a] -> Vector a

-- | O(n). Convert list with known length.
fromListN :: Int -> [a] -> Vector a

-- | O(1). Length of vector.
length :: Vector a -> Int

-- | O(1). Test for empty.
null :: Vector a -> Bool

-- | O(1). Index (unsafe).
(!) :: Vector a -> Int -> a

-- | O(1). Safe indexing.
(!?) :: Vector a -> Int -> Maybe a

-- | O(1). First element (unsafe).
head :: Vector a -> a

-- | O(1). Last element (unsafe).
last :: Vector a -> a

-- | O(1). Unsafe index.
unsafeIndex :: Vector a -> Int -> a

-- | O(1). Unsafe head.
unsafeHead :: Vector a -> a

-- | O(1). Unsafe last.
unsafeLast :: Vector a -> a

-- | O(1). Slice from offset with length (view).
slice :: Int -> Int -> Vector a -> Vector a

-- | O(1). All but last element (view).
init :: Vector a -> Vector a

-- | O(1). All but first element (view).
tail :: Vector a -> Vector a

-- | O(1). First n elements (view).
take :: Int -> Vector a -> Vector a

-- | O(1). Drop first n elements (view).
drop :: Int -> Vector a -> Vector a

-- | O(1). Split at position.
splitAt :: Int -> Vector a -> (Vector a, Vector a)

-- | O(n). Prepend element.
cons :: a -> Vector a -> Vector a

-- | O(n). Append element.
snoc :: Vector a -> a -> Vector a

-- | O(n). Append vectors.
(++) :: Vector a -> Vector a -> Vector a

-- | O(n). Concatenate vectors.
concat :: [Vector a] -> Vector a

-- | O(n). Force evaluation and compaction.
force :: Vector a -> Vector a

-- | O(n). Map function over elements.
map :: (a -> b) -> Vector a -> Vector b

-- | O(n). Map with index.
imap :: (Int -> a -> b) -> Vector a -> Vector b

-- | O(n). Map and concatenate.
concatMap :: (a -> Vector b) -> Vector a -> Vector b

-- | O(n). Monadic map.
mapM :: Monad m => (a -> m b) -> Vector a -> m (Vector b)

-- | O(n). Monadic map, discarding results.
mapM_ :: Monad m => (a -> m b) -> Vector a -> m ()

-- | O(n). Flipped monadic map.
forM :: Monad m => Vector a -> (a -> m b) -> m (Vector b)

-- | O(n). Flipped monadic map, discarding results.
forM_ :: Monad m => Vector a -> (a -> m b) -> m ()

-- | O(n). Zip with function.
zipWith :: (a -> b -> c) -> Vector a -> Vector b -> Vector c

-- | O(n). Zip three vectors.
zipWith3 :: (a -> b -> c -> d) -> Vector a -> Vector b -> Vector c -> Vector d

-- | O(n). Zip four vectors.
zipWith4 :: (a -> b -> c -> d -> e) -> Vector a -> Vector b -> Vector c -> Vector d -> Vector e

-- | O(n). Monadic zipWith.
zipWithM :: Monad m => (a -> b -> m c) -> Vector a -> Vector b -> m (Vector c)

-- | O(n). Monadic zipWith, discarding results.
zipWithM_ :: Monad m => (a -> b -> m c) -> Vector a -> Vector b -> m ()

-- | O(n). Zip to pairs.
zip :: Vector a -> Vector b -> Vector (a, b)

-- | O(n). Zip three vectors.
zip3 :: Vector a -> Vector b -> Vector c -> Vector (a, b, c)

-- | O(n). Zip four vectors.
zip4 :: Vector a -> Vector b -> Vector c -> Vector d -> Vector (a, b, c, d)

-- | O(n). Unzip pairs.
unzip :: Vector (a, b) -> (Vector a, Vector b)

-- | O(n). Unzip triples.
unzip3 :: Vector (a, b, c) -> (Vector a, Vector b, Vector c)

-- | O(n). Unzip quadruples.
unzip4 :: Vector (a, b, c, d) -> (Vector a, Vector b, Vector c, Vector d)

-- | O(n). Filter by predicate.
filter :: (a -> Bool) -> Vector a -> Vector a

-- | O(n). Filter with index.
ifilter :: (Int -> a -> Bool) -> Vector a -> Vector a

-- | O(n). Monadic filter.
filterM :: Monad m => (a -> m Bool) -> Vector a -> m (Vector a)

-- | O(n). Take while predicate holds.
takeWhile :: (a -> Bool) -> Vector a -> Vector a

-- | O(n). Drop while predicate holds.
dropWhile :: (a -> Bool) -> Vector a -> Vector a

-- | O(n). Partition by predicate.
partition :: (a -> Bool) -> Vector a -> (Vector a, Vector a)

-- | O(n). Span while predicate holds.
span :: (a -> Bool) -> Vector a -> (Vector a, Vector a)

-- | O(n). Break at first failure.
break :: (a -> Bool) -> Vector a -> (Vector a, Vector a)

-- | O(n). Test membership.
elem :: Eq a => a -> Vector a -> Bool

-- | O(n). Test non-membership.
notElem :: Eq a => a -> Vector a -> Bool

-- | O(n). Find first match.
find :: (a -> Bool) -> Vector a -> Maybe a

-- | O(n). Find index of first match.
findIndex :: (a -> Bool) -> Vector a -> Maybe Int

-- | O(n). Find all matching indices.
findIndices :: (a -> Bool) -> Vector a -> Vector Int

-- | O(n). Find index of element.
elemIndex :: Eq a => a -> Vector a -> Maybe Int

-- | O(n). Find all indices of element.
elemIndices :: Eq a => a -> Vector a -> Vector Int

-- | O(n). Left fold.
foldl :: (b -> a -> b) -> b -> Vector a -> b

-- | O(n). Strict left fold.
foldl' :: (b -> a -> b) -> b -> Vector a -> b

-- | O(n). Left fold without starting value.
foldl1 :: (a -> a -> a) -> Vector a -> a

-- | O(n). Strict left fold without starting value.
foldl1' :: (a -> a -> a) -> Vector a -> a

-- | O(n). Right fold.
foldr :: (a -> b -> b) -> b -> Vector a -> b

-- | O(n). Strict right fold.
foldr' :: (a -> b -> b) -> b -> Vector a -> b

-- | O(n). Right fold without starting value.
foldr1 :: (a -> a -> a) -> Vector a -> a

-- | O(n). Strict right fold without starting value.
foldr1' :: (a -> a -> a) -> Vector a -> a

-- | O(n). Indexed left fold.
ifoldl :: (b -> Int -> a -> b) -> b -> Vector a -> b

-- | O(n). Strict indexed left fold.
ifoldl' :: (b -> Int -> a -> b) -> b -> Vector a -> b

-- | O(n). Indexed right fold.
ifoldr :: (Int -> a -> b -> b) -> b -> Vector a -> b

-- | O(n). Strict indexed right fold.
ifoldr' :: (Int -> a -> b -> b) -> b -> Vector a -> b

-- | O(n). All elements satisfy predicate.
all :: (a -> Bool) -> Vector a -> Bool

-- | O(n). Any element satisfies predicate.
any :: (a -> Bool) -> Vector a -> Bool

-- | O(n). Sum of elements.
sum :: Num a => Vector a -> a

-- | O(n). Product of elements.
product :: Num a => Vector a -> a

-- | O(n). Maximum element.
maximum :: Ord a => Vector a -> a

-- | O(n). Minimum element.
minimum :: Ord a => Vector a -> a

-- | O(n). Maximum by comparison function.
maximumBy :: (a -> a -> Ordering) -> Vector a -> a

-- | O(n). Minimum by comparison function.
minimumBy :: (a -> a -> Ordering) -> Vector a -> a

-- | O(n). Index of maximum element.
maxIndex :: Ord a => Vector a -> Int

-- | O(n). Index of minimum element.
minIndex :: Ord a => Vector a -> Int

-- | O(n). Prefix scan (exclusive).
prescanl :: (a -> b -> a) -> a -> Vector b -> Vector a

-- | O(n). Strict prefix scan (exclusive).
prescanl' :: (a -> b -> a) -> a -> Vector b -> Vector a

-- | O(n). Suffix scan (exclusive).
postscanl :: (a -> b -> a) -> a -> Vector b -> Vector a

-- | O(n). Strict suffix scan (exclusive).
postscanl' :: (a -> b -> a) -> a -> Vector b -> Vector a

-- | O(n). Left scan (inclusive).
scanl :: (a -> b -> a) -> a -> Vector b -> Vector a

-- | O(n). Strict left scan (inclusive).
scanl' :: (a -> b -> a) -> a -> Vector b -> Vector a

-- | O(n). Left scan without starting value.
scanl1 :: (a -> a -> a) -> Vector a -> Vector a

-- | O(n). Strict left scan without starting value.
scanl1' :: (a -> a -> a) -> Vector a -> Vector a

-- | O(n). Right prefix scan.
prescanr :: (a -> b -> b) -> b -> Vector a -> Vector b

-- | O(n). Strict right prefix scan.
prescanr' :: (a -> b -> b) -> b -> Vector a -> Vector b

-- | O(n). Right suffix scan.
postscanr :: (a -> b -> b) -> b -> Vector a -> Vector b

-- | O(n). Strict right suffix scan.
postscanr' :: (a -> b -> b) -> b -> Vector a -> Vector b

-- | O(n). Right scan (inclusive).
scanr :: (a -> b -> b) -> b -> Vector a -> Vector b

-- | O(n). Strict right scan (inclusive).
scanr' :: (a -> b -> b) -> b -> Vector a -> Vector b

-- | O(n). Right scan without starting value.
scanr1 :: (a -> a -> a) -> Vector a -> Vector a

-- | O(n). Strict right scan without starting value.
scanr1' :: (a -> a -> a) -> Vector a -> Vector a

-- | O(n). Convert to list.
toList :: Vector a -> [a]

-- | O(n). Convert between vector types.
convert :: (Vector a -> Vector b)

-- | O(n). Allocate new mutable vector.
new :: Int -> IO (MVector RealWorld a)

-- | O(n). Allocate with initial value.
newWith :: Int -> a -> IO (MVector RealWorld a)

-- | O(1). Read element.
read :: MVector s a -> Int -> ST s a

-- | O(1). Write element.
write :: MVector s a -> Int -> a -> ST s ()

-- | O(1). Modify element.
modify :: MVector s a -> (a -> a) -> Int -> ST s ()

-- | O(1). Swap two elements.
swap :: MVector s a -> Int -> Int -> ST s ()

-- | O(n). Grow vector capacity.
grow :: MVector s a -> Int -> ST s (MVector s a)

-- | O(n). Freeze mutable to immutable.
freeze :: MVector s a -> ST s (Vector a)

-- | O(n). Thaw immutable to mutable.
thaw :: Vector a -> ST s (MVector s a)

-- | O(n). Copy between mutable vectors.
copy :: MVector s a -> MVector s a -> ST s ()

-- | O(1). Unsafe freeze (no copy).
unsafeFreeze :: MVector s a -> ST s (Vector a)

-- | O(1). Unsafe thaw (no copy).
unsafeThaw :: Vector a -> ST s (MVector s a)

-- | O(n log n). Sort elements.
sort :: Ord a => Vector a -> Vector a

-- | O(n log n). Sort by comparison function.
sortBy :: (a -> a -> Ordering) -> Vector a -> Vector a

-- | O(n log n). Sort by key extraction.
sortOn :: Ord b => (a -> b) -> Vector a -> Vector a

-- This is a specification file.
-- Actual implementation provided by the compiler.
