-- |
-- Module      : BHC.Numeric.Vector
-- Description : Dense numeric vectors
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Unboxed numeric vectors with high-performance operations.
-- Vectors are 1-dimensional tensors with optimized operations.

{-# LANGUAGE BangPatterns #-}

module BHC.Numeric.Vector (
    -- * Vector type
    Vector,

    -- * Construction
    empty, singleton,
    replicate, generate,
    fromList, fromListN,
    enumFromN, enumFromStepN,

    -- * Basic operations
    length, null,
    (!), (!?),
    head, last,
    tail, init,

    -- * Slicing
    slice, take, drop,
    splitAt, takeWhile, dropWhile,

    -- * Construction from vectors
    cons, snoc,
    (++), concat,

    -- * Element-wise operations
    map, imap,
    zipWith, zipWith3,
    izipWith, izipWith3,

    -- * Folds
    foldl, foldl', foldl1, foldl1',
    foldr, foldr1,
    ifoldl, ifoldl', ifoldr,

    -- * Specialized folds
    sum, product,
    maximum, minimum,
    maximumBy, minimumBy,
    all, any,

    -- * Scans
    prescanl, prescanl',
    postscanl, postscanl',
    scanl, scanl', scanl1, scanl1',

    -- * Search
    elem, notElem,
    find, findIndex,
    elemIndex, elemIndices,

    -- * Numeric operations
    dot, norm, normalize,
    add, sub, mul, div,
    scale, negate, abs,

    -- * Sorting
    sort, sortBy,
    indexed,

    -- * Conversion
    toList,
    convert,

    -- * Unboxed vectors
    UVector,
) where

import BHC.Prelude hiding (
    length, null, head, last, tail, init,
    take, drop, splitAt, takeWhile, dropWhile,
    (++), concat, map, zipWith, zipWith3,
    foldl, foldl1, foldr, foldr1,
    sum, product, maximum, minimum,
    all, any, elem, notElem,
    replicate, negate, abs
    )
import qualified BHC.Prelude as P

-- ============================================================
-- Vector Type
-- ============================================================

-- | A contiguous array of elements.
data Vector a = Vector
    { vecData   :: !VectorData
    , vecOffset :: !Int
    , vecLength :: !Int
    }

-- | Internal vector storage.
data VectorData

-- | Unboxed vector (no pointer indirection).
data UVector a = UVector
    { uvecData   :: !UVectorData
    , uvecOffset :: !Int
    , uvecLength :: !Int
    }

data UVectorData

-- ============================================================
-- Construction
-- ============================================================

-- | Empty vector.
empty :: Vector a
empty = Vector undefined 0 0

-- | Single element vector.
singleton :: a -> Vector a
singleton x = fromList [x]

-- | Vector of @n@ copies of element.
replicate :: Int -> a -> Vector a
replicate n x = fromList (P.replicate n x)

-- | Generate vector using function.
--
-- >>> generate 5 (*2)
-- [0, 2, 4, 6, 8]
generate :: Int -> (Int -> a) -> Vector a
generate n f = fromList [f i | i <- [0..n-1]]

-- | Create vector from list.
fromList :: [a] -> Vector a
fromList = undefined

-- | Create vector from list with known length.
fromListN :: Int -> [a] -> Vector a
fromListN = undefined

-- | Enumerate from starting value.
--
-- >>> enumFromN 5 3
-- [5, 6, 7]
enumFromN :: Num a => a -> Int -> Vector a
enumFromN start n = generate n (\i -> start + P.fromIntegral i)

-- | Enumerate with step.
--
-- >>> enumFromStepN 0 2 5
-- [0, 2, 4, 6, 8]
enumFromStepN :: Num a => a -> a -> Int -> Vector a
enumFromStepN start step n = generate n (\i -> start + step * P.fromIntegral i)

-- ============================================================
-- Basic Operations
-- ============================================================

-- | Length of vector.
length :: Vector a -> Int
length = vecLength

-- | Is the vector empty?
null :: Vector a -> Bool
null v = length v == 0

-- | Index into vector (unsafe).
(!) :: Vector a -> Int -> a
(!) = undefined

-- | Index into vector (safe).
(!?) :: Vector a -> Int -> Maybe a
v !? i
    | i < 0 || i >= length v = Nothing
    | otherwise = Just (v ! i)

-- | First element (unsafe).
head :: Vector a -> a
head v = v ! 0

-- | Last element (unsafe).
last :: Vector a -> a
last v = v ! (length v - 1)

-- | All elements except first.
tail :: Vector a -> Vector a
tail v = slice 1 (length v - 1) v

-- | All elements except last.
init :: Vector a -> Vector a
init v = slice 0 (length v - 1) v

-- ============================================================
-- Slicing
-- ============================================================

-- | Extract slice starting at index with given length.
slice :: Int -> Int -> Vector a -> Vector a
slice start len v = v
    { vecOffset = vecOffset v + start
    , vecLength = len
    }

-- | Take first n elements.
take :: Int -> Vector a -> Vector a
take n v = slice 0 (P.min n (length v)) v

-- | Drop first n elements.
drop :: Int -> Vector a -> Vector a
drop n v = slice n (P.max 0 (length v - n)) v

-- | Split at index.
splitAt :: Int -> Vector a -> (Vector a, Vector a)
splitAt n v = (take n v, drop n v)

-- | Take while predicate holds.
takeWhile :: (a -> Bool) -> Vector a -> Vector a
takeWhile p v = case findIndex (P.not . p) v of
    Nothing -> v
    Just i  -> take i v

-- | Drop while predicate holds.
dropWhile :: (a -> Bool) -> Vector a -> Vector a
dropWhile p v = case findIndex (P.not . p) v of
    Nothing -> empty
    Just i  -> drop i v

-- ============================================================
-- Construction from Vectors
-- ============================================================

-- | Prepend element.
cons :: a -> Vector a -> Vector a
cons x v = fromList (x : toList v)

-- | Append element.
snoc :: Vector a -> a -> Vector a
snoc v x = fromList (toList v P.++ [x])

-- | Concatenate two vectors.
(++) :: Vector a -> Vector a -> Vector a
v1 ++ v2 = fromList (toList v1 P.++ toList v2)

-- | Concatenate list of vectors.
concat :: [Vector a] -> Vector a
concat vs = fromList (P.concatMap toList vs)

-- ============================================================
-- Element-wise Operations
-- ============================================================

-- | Map function over elements.
map :: (a -> b) -> Vector a -> Vector b
map f v = generate (length v) (\i -> f (v ! i))

-- | Map with index.
imap :: (Int -> a -> b) -> Vector a -> Vector b
imap f v = generate (length v) (\i -> f i (v ! i))

-- | Zip two vectors with function.
zipWith :: (a -> b -> c) -> Vector a -> Vector b -> Vector c
zipWith f va vb =
    let n = P.min (length va) (length vb)
    in generate n (\i -> f (va ! i) (vb ! i))

-- | Zip three vectors with function.
zipWith3 :: (a -> b -> c -> d) -> Vector a -> Vector b -> Vector c -> Vector d
zipWith3 f va vb vc =
    let n = P.minimum [length va, length vb, length vc]
    in generate n (\i -> f (va ! i) (vb ! i) (vc ! i))

-- | Zip with index.
izipWith :: (Int -> a -> b -> c) -> Vector a -> Vector b -> Vector c
izipWith f va vb =
    let n = P.min (length va) (length vb)
    in generate n (\i -> f i (va ! i) (vb ! i))

-- | Zip three with index.
izipWith3 :: (Int -> a -> b -> c -> d) -> Vector a -> Vector b -> Vector c -> Vector d
izipWith3 f va vb vc =
    let n = P.minimum [length va, length vb, length vc]
    in generate n (\i -> f i (va ! i) (vb ! i) (vc ! i))

-- ============================================================
-- Folds
-- ============================================================

-- | Left fold.
foldl :: (b -> a -> b) -> b -> Vector a -> b
foldl f z v = go 0 z
  where
    n = length v
    go !i !acc
        | i >= n    = acc
        | otherwise = go (i + 1) (f acc (v ! i))

-- | Strict left fold.
foldl' :: (b -> a -> b) -> b -> Vector a -> b
foldl' = foldl  -- Already strict due to bang patterns

-- | Left fold without starting value (unsafe).
foldl1 :: (a -> a -> a) -> Vector a -> a
foldl1 f v = foldl f (head v) (tail v)

-- | Strict left fold without starting value.
foldl1' :: (a -> a -> a) -> Vector a -> a
foldl1' = foldl1

-- | Right fold.
foldr :: (a -> b -> b) -> b -> Vector a -> b
foldr f z v = go (length v - 1)
  where
    go i
        | i < 0     = z
        | otherwise = f (v ! i) (go (i - 1))

-- | Right fold without starting value (unsafe).
foldr1 :: (a -> a -> a) -> Vector a -> a
foldr1 f v = foldr f (last v) (init v)

-- | Left fold with index.
ifoldl :: (b -> Int -> a -> b) -> b -> Vector a -> b
ifoldl f z v = go 0 z
  where
    n = length v
    go !i !acc
        | i >= n    = acc
        | otherwise = go (i + 1) (f acc i (v ! i))

-- | Strict left fold with index.
ifoldl' :: (b -> Int -> a -> b) -> b -> Vector a -> b
ifoldl' = ifoldl

-- | Right fold with index.
ifoldr :: (Int -> a -> b -> b) -> b -> Vector a -> b
ifoldr f z v = go 0
  where
    n = length v
    go i
        | i >= n    = z
        | otherwise = f i (v ! i) (go (i + 1))

-- ============================================================
-- Specialized Folds
-- ============================================================

-- | Sum of elements.
sum :: Num a => Vector a -> a
sum = foldl' (+) 0

-- | Product of elements.
product :: Num a => Vector a -> a
product = foldl' (*) 1

-- | Maximum element (unsafe on empty).
maximum :: Ord a => Vector a -> a
maximum = foldl1' P.max

-- | Minimum element (unsafe on empty).
minimum :: Ord a => Vector a -> a
minimum = foldl1' P.min

-- | Maximum by comparison function.
maximumBy :: (a -> a -> Ordering) -> Vector a -> a
maximumBy cmp = foldl1' (\a b -> if cmp a b == GT then a else b)

-- | Minimum by comparison function.
minimumBy :: (a -> a -> Ordering) -> Vector a -> a
minimumBy cmp = foldl1' (\a b -> if cmp a b == LT then a else b)

-- | All elements satisfy predicate.
all :: (a -> Bool) -> Vector a -> Bool
all p = foldl' (\acc x -> acc P.&& p x) True

-- | Any element satisfies predicate.
any :: (a -> Bool) -> Vector a -> Bool
any p = foldl' (\acc x -> acc P.|| p x) False

-- ============================================================
-- Scans
-- ============================================================

-- | Prefix scan (exclusive).
prescanl :: (a -> b -> a) -> a -> Vector b -> Vector a
prescanl f z v = generate (length v) (\i ->
    foldl' f z (take i v))

-- | Strict prefix scan.
prescanl' :: (a -> b -> a) -> a -> Vector b -> Vector a
prescanl' = prescanl

-- | Postfix scan (inclusive).
postscanl :: (a -> b -> a) -> a -> Vector b -> Vector a
postscanl f z v = generate (length v) (\i ->
    foldl' f z (take (i + 1) v))

-- | Strict postfix scan.
postscanl' :: (a -> b -> a) -> a -> Vector b -> Vector a
postscanl' = postscanl

-- | Scan (like scanl but returns vector).
scanl :: (a -> b -> a) -> a -> Vector b -> Vector a
scanl = prescanl

-- | Strict scan.
scanl' :: (a -> b -> a) -> a -> Vector b -> Vector a
scanl' = prescanl'

-- | Scan without starting value.
scanl1 :: (a -> a -> a) -> Vector a -> Vector a
scanl1 f v = postscanl f (head v) (tail v)

-- | Strict scan without starting value.
scanl1' :: (a -> a -> a) -> Vector a -> Vector a
scanl1' = scanl1

-- ============================================================
-- Search
-- ============================================================

-- | Is element in vector?
elem :: Eq a => a -> Vector a -> Bool
elem x = any (== x)

-- | Is element not in vector?
notElem :: Eq a => a -> Vector a -> Bool
notElem x = P.not . elem x

-- | Find first element satisfying predicate.
find :: (a -> Bool) -> Vector a -> Maybe a
find p v = case findIndex p v of
    Nothing -> Nothing
    Just i  -> Just (v ! i)

-- | Find index of first element satisfying predicate.
findIndex :: (a -> Bool) -> Vector a -> Maybe Int
findIndex p v = go 0
  where
    n = length v
    go i
        | i >= n    = Nothing
        | p (v ! i) = Just i
        | otherwise = go (i + 1)

-- | Find index of element.
elemIndex :: Eq a => a -> Vector a -> Maybe Int
elemIndex x = findIndex (== x)

-- | Find all indices of element.
elemIndices :: Eq a => a -> Vector a -> Vector Int
elemIndices x v = fromList [i | i <- [0..length v - 1], v ! i == x]

-- ============================================================
-- Numeric Operations
-- ============================================================

-- | Dot product of two vectors.
--
-- >>> dot [1, 2, 3] [4, 5, 6]
-- 32
foreign import ccall "bhc_vector_dot"
    dot :: Num a => Vector a -> Vector a -> a

-- | Euclidean norm (L2).
norm :: Floating a => Vector a -> a
norm v = P.sqrt (dot v v)

-- | Normalize to unit length.
normalize :: Floating a => Vector a -> Vector a
normalize v = scale (1 / norm v) v

-- | Element-wise addition.
add :: Num a => Vector a -> Vector a -> Vector a
add = zipWith (+)

-- | Element-wise subtraction.
sub :: Num a => Vector a -> Vector a -> Vector a
sub = zipWith (-)

-- | Element-wise multiplication.
mul :: Num a => Vector a -> Vector a -> Vector a
mul = zipWith (*)

-- | Element-wise division.
div :: Fractional a => Vector a -> Vector a -> Vector a
div = zipWith (/)

-- | Scale vector by scalar.
scale :: Num a => a -> Vector a -> Vector a
scale k = map (* k)

-- | Negate all elements.
negate :: Num a => Vector a -> Vector a
negate = map P.negate

-- | Absolute value of all elements.
abs :: Num a => Vector a -> Vector a
abs = map P.abs

-- ============================================================
-- Sorting
-- ============================================================

-- | Sort in ascending order.
sort :: Ord a => Vector a -> Vector a
sort v = fromList (P.sort (toList v))

-- | Sort by comparison function.
sortBy :: (a -> a -> Ordering) -> Vector a -> Vector a
sortBy cmp v = fromList (P.sortBy cmp (toList v))

-- | Pair each element with its index.
indexed :: Vector a -> Vector (Int, a)
indexed v = generate (length v) (\i -> (i, v ! i))

-- ============================================================
-- Conversion
-- ============================================================

-- | Convert to list.
toList :: Vector a -> [a]
toList v = [v ! i | i <- [0..length v - 1]]

-- | Convert between vector types.
convert :: Vector a -> UVector a
convert = undefined
