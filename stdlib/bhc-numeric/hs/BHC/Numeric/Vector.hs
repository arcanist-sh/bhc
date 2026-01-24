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
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}

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

    -- * Modification
    update, filter,

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

    -- * Type class
    VectorElem,
) where

import BHC.Prelude hiding (
    length, null, head, last, tail, init,
    take, drop, splitAt, takeWhile, dropWhile,
    (++), concat, map, zipWith, zipWith3,
    foldl, foldl1, foldr, foldr1,
    sum, product, maximum, minimum,
    all, any, elem, notElem, filter,
    replicate, negate, abs
    )
import qualified BHC.Prelude as P
import Foreign.Ptr (Ptr, nullPtr)
import Foreign.ForeignPtr (ForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Array (peekArray, withArrayLen)
import Foreign.Storable (Storable, sizeOf, peek, poke)
import System.IO.Unsafe (unsafePerformIO)

-- FFI imports for f64 vectors
foreign import ccall unsafe "bhc_vector_new_f64"
    c_vector_new_f64 :: Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_from_f64"
    c_vector_from_f64 :: Ptr Double -> Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_free_f64"
    c_vector_free_f64 :: Ptr VectorData -> IO ()

foreign import ccall unsafe "&bhc_vector_free_f64"
    c_vector_finalizer_f64 :: FunPtr (Ptr VectorData -> IO ())

foreign import ccall unsafe "bhc_vector_get_f64"
    c_vector_get_f64 :: Ptr VectorData -> Int -> IO Double

foreign import ccall unsafe "bhc_vector_len_f64"
    c_vector_len_f64 :: Ptr VectorData -> IO Int

foreign import ccall unsafe "bhc_vector_dot_f64"
    c_vector_dot_f64 :: Ptr VectorData -> Ptr VectorData -> IO Double

foreign import ccall unsafe "bhc_vector_sum_f64"
    c_vector_sum_f64 :: Ptr VectorData -> IO Double

foreign import ccall unsafe "bhc_vector_norm_f64"
    c_vector_norm_f64 :: Ptr VectorData -> IO Double

-- FFI imports for f32 vectors
foreign import ccall unsafe "bhc_vector_new_f32"
    c_vector_new_f32 :: Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_from_f32"
    c_vector_from_f32 :: Ptr Float -> Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_free_f32"
    c_vector_free_f32 :: Ptr VectorData -> IO ()

foreign import ccall unsafe "bhc_vector_get_f32"
    c_vector_get_f32 :: Ptr VectorData -> Int -> IO Float

foreign import ccall unsafe "bhc_vector_len_f32"
    c_vector_len_f32 :: Ptr VectorData -> IO Int

foreign import ccall unsafe "bhc_vector_dot_f32"
    c_vector_dot_f32 :: Ptr VectorData -> Ptr VectorData -> IO Float

foreign import ccall unsafe "bhc_vector_sum_f32"
    c_vector_sum_f32 :: Ptr VectorData -> IO Float

-- FFI imports for i64 vectors (Int)
foreign import ccall unsafe "bhc_vector_new_i64"
    c_vector_new_i64 :: Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_from_i64"
    c_vector_from_i64 :: Ptr Int -> Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_get_i64"
    c_vector_get_i64 :: Ptr VectorData -> Int -> IO Int

foreign import ccall unsafe "bhc_vector_len_i64"
    c_vector_len_i64 :: Ptr VectorData -> IO Int

foreign import ccall unsafe "bhc_vector_sum_i64"
    c_vector_sum_i64 :: Ptr VectorData -> IO Int

-- ============================================================
-- Vector Type
-- ============================================================

-- | A contiguous array of elements.
-- Uses foreign pointer to Rust-managed memory.
data Vector a = Vector
    { vecPtr    :: !(ForeignPtr VectorData)
    , vecOffset :: !Int
    , vecLength :: !Int
    }

-- | Internal vector storage (opaque Rust type).
data VectorData

-- | Unboxed vector (no pointer indirection).
-- For now, same as Vector but may have specialized representation.
data UVector a = UVector
    { uvecPtr    :: !(ForeignPtr VectorData)
    , uvecOffset :: !Int
    , uvecLength :: !Int
    }

-- | Type class for vector element operations
class VectorElem a where
    vectorFromList :: [a] -> IO (Vector a)
    vectorGet :: Vector a -> Int -> IO a
    vectorLen :: Vector a -> IO Int
    vectorDot :: Vector a -> Vector a -> IO a
    vectorSum :: Vector a -> IO a

instance VectorElem Double where
    vectorFromList xs = do
        withArrayLen xs $ \len ptr -> do
            vptr <- c_vector_from_f64 ptr len
            fp <- newForeignPtr c_vector_finalizer_f64 vptr
            return $ Vector fp 0 len
    vectorGet (Vector fp off _) i = withForeignPtr fp $ \ptr ->
        c_vector_get_f64 ptr (off + i)
    vectorLen (Vector fp _ _) = withForeignPtr fp c_vector_len_f64
    vectorDot (Vector fp1 _ _) (Vector fp2 _ _) =
        withForeignPtr fp1 $ \p1 ->
        withForeignPtr fp2 $ \p2 ->
            c_vector_dot_f64 p1 p2
    vectorSum (Vector fp _ _) = withForeignPtr fp c_vector_sum_f64

instance VectorElem Float where
    vectorFromList xs = do
        withArrayLen xs $ \len ptr -> do
            vptr <- c_vector_from_f32 ptr len
            fp <- newForeignPtr c_vector_finalizer_f64 vptr  -- Uses same finalizer shape
            return $ Vector fp 0 len
    vectorGet (Vector fp off _) i = withForeignPtr fp $ \ptr ->
        c_vector_get_f32 ptr (off + i)
    vectorLen (Vector fp _ _) = withForeignPtr fp c_vector_len_f32
    vectorDot (Vector fp1 _ _) (Vector fp2 _ _) =
        withForeignPtr fp1 $ \p1 ->
        withForeignPtr fp2 $ \p2 ->
            c_vector_dot_f32 p1 p2
    vectorSum (Vector fp _ _) = withForeignPtr fp c_vector_sum_f32

instance VectorElem Int where
    vectorFromList xs = do
        withArrayLen xs $ \len ptr -> do
            vptr <- c_vector_from_i64 ptr len
            fp <- newForeignPtr c_vector_finalizer_f64 vptr  -- Uses same finalizer shape
            return $ Vector fp 0 len
    vectorGet (Vector fp off _) i = withForeignPtr fp $ \ptr ->
        c_vector_get_i64 ptr (off + i)
    vectorLen (Vector fp _ _) = withForeignPtr fp c_vector_len_i64
    vectorDot v1 v2 = foldl' (+) 0 (zipWith (*) v1 v2)  -- Pure implementation for Int
    vectorSum (Vector fp _ _) = withForeignPtr fp c_vector_sum_i64

-- ============================================================
-- Construction
-- ============================================================

-- | /O(1)/. The empty vector.
--
-- >>> length empty
-- 0
empty :: Vector a
empty = Vector undefined 0 0

-- | /O(n)/. A vector with a single element.
--
-- >>> singleton 42
-- [42]
singleton :: VectorElem a => a -> Vector a
singleton x = fromList [x]

-- | /O(n)/. A vector of @n@ copies of the given element.
--
-- >>> replicate 5 3.14
-- [3.14, 3.14, 3.14, 3.14, 3.14]
replicate :: VectorElem a => Int -> a -> Vector a
replicate n x = fromList (P.replicate n x)

-- | /O(n)/. Generate a vector by applying a function to each index.
--
-- >>> generate 5 (*2)
-- [0, 2, 4, 6, 8]
--
-- >>> generate 4 (\i -> i * i)
-- [0, 1, 4, 9]
generate :: VectorElem a => Int -> (Int -> a) -> Vector a
generate n f = fromList [f i | i <- [0..n-1]]

-- | /O(n)/. Create a vector from a list.
--
-- >>> fromList [1, 2, 3, 4, 5]
-- [1, 2, 3, 4, 5]
fromList :: VectorElem a => [a] -> Vector a
fromList xs = unsafePerformIO $ vectorFromList xs
{-# NOINLINE fromList #-}

-- | /O(n)/. Create a vector from the first @n@ elements of a list.
--
-- >>> fromListN 3 [1, 2, 3, 4, 5]
-- [1, 2, 3]
fromListN :: VectorElem a => Int -> [a] -> Vector a
fromListN n xs = fromList (P.take n xs)

-- | /O(n)/. Enumerate @n@ values starting from a given value.
--
-- >>> enumFromN 5 3
-- [5, 6, 7]
--
-- >>> enumFromN 0.0 4
-- [0.0, 1.0, 2.0, 3.0]
enumFromN :: (Num a, VectorElem a) => a -> Int -> Vector a
enumFromN start n = generate n (\i -> start + P.fromIntegral i)

-- | /O(n)/. Enumerate @n@ values starting from a given value with a step.
--
-- >>> enumFromStepN 0 2 5
-- [0, 2, 4, 6, 8]
--
-- >>> enumFromStepN 10 (-1) 5
-- [10, 9, 8, 7, 6]
enumFromStepN :: (Num a, VectorElem a) => a -> a -> Int -> Vector a
enumFromStepN start step n = generate n (\i -> start + step * P.fromIntegral i)

-- ============================================================
-- Basic Operations
-- ============================================================

-- | /O(1)/. The number of elements in the vector.
--
-- >>> length (fromList [1, 2, 3, 4, 5])
-- 5
length :: Vector a -> Int
length = vecLength

-- | /O(1)/. Test whether a vector is empty.
--
-- >>> null empty
-- True
-- >>> null (singleton 1)
-- False
null :: Vector a -> Bool
null v = length v == 0

-- | /O(1)/. Index into a vector (unsafe).
--
-- __Warning__: Throws an error if the index is out of bounds.
--
-- >>> fromList [10, 20, 30] ! 1
-- 20
(!) :: VectorElem a => Vector a -> Int -> a
(!) v i = unsafePerformIO $ vectorGet v i
{-# NOINLINE (!) #-}

-- | /O(1)/. Safe indexing into a vector.
--
-- >>> fromList [10, 20, 30] !? 1
-- Just 20
-- >>> fromList [10, 20, 30] !? 5
-- Nothing
(!?) :: VectorElem a => Vector a -> Int -> Maybe a
v !? i
    | i < 0 || i >= length v = Nothing
    | otherwise = Just (v ! i)

-- | /O(1)/. The first element of a vector.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> head (fromList [1, 2, 3])
-- 1
head :: VectorElem a => Vector a -> a
head v = v ! 0

-- | /O(1)/. The last element of a vector.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> last (fromList [1, 2, 3])
-- 3
last :: VectorElem a => Vector a -> a
last v = v ! (length v - 1)

-- | /O(1)/. All elements except the first (a view, no copy).
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> toList (tail (fromList [1, 2, 3, 4]))
-- [2, 3, 4]
tail :: VectorElem a => Vector a -> Vector a
tail v = slice 1 (length v - 1) v

-- | /O(1)/. All elements except the last (a view, no copy).
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> toList (init (fromList [1, 2, 3, 4]))
-- [1, 2, 3]
init :: VectorElem a => Vector a -> Vector a
init v = slice 0 (length v - 1) v

-- ============================================================
-- Slicing
-- ============================================================

-- | /O(1)/. Extract a slice starting at an index with the given length.
-- Returns a view into the original vector (no copy).
--
-- >>> toList (slice 1 3 (fromList [0, 1, 2, 3, 4, 5]))
-- [1, 2, 3]
slice :: Int -> Int -> Vector a -> Vector a
slice start len v = v
    { vecOffset = vecOffset v + start
    , vecLength = len
    }

-- | /O(1)/. Take the first @n@ elements (a view, no copy).
--
-- >>> toList (take 3 (fromList [1, 2, 3, 4, 5]))
-- [1, 2, 3]
-- >>> toList (take 10 (fromList [1, 2, 3]))
-- [1, 2, 3]
take :: Int -> Vector a -> Vector a
take n v = slice 0 (P.min n (length v)) v

-- | /O(1)/. Drop the first @n@ elements (a view, no copy).
--
-- >>> toList (drop 2 (fromList [1, 2, 3, 4, 5]))
-- [3, 4, 5]
-- >>> toList (drop 10 (fromList [1, 2, 3]))
-- []
drop :: Int -> Vector a -> Vector a
drop n v = slice n (P.max 0 (length v - n)) v

-- | /O(1)/. Split a vector at the given index.
--
-- >>> let (a, b) = splitAt 2 (fromList [1, 2, 3, 4, 5])
-- >>> (toList a, toList b)
-- ([1, 2], [3, 4, 5])
splitAt :: Int -> Vector a -> (Vector a, Vector a)
splitAt n v = (take n v, drop n v)

-- | /O(n)/. Take elements while the predicate holds.
--
-- >>> toList (takeWhile (< 5) (fromList [1, 3, 5, 7, 9]))
-- [1, 3]
takeWhile :: VectorElem a => (a -> Bool) -> Vector a -> Vector a
takeWhile p v = case findIndex (P.not . p) v of
    Nothing -> v
    Just i  -> take i v

-- | /O(n)/. Drop elements while the predicate holds.
--
-- >>> toList (dropWhile (< 5) (fromList [1, 3, 5, 7, 9]))
-- [5, 7, 9]
dropWhile :: VectorElem a => (a -> Bool) -> Vector a -> Vector a
dropWhile p v = case findIndex (P.not . p) v of
    Nothing -> empty
    Just i  -> drop i v

-- ============================================================
-- Construction from Vectors
-- ============================================================

-- | /O(n)/. Prepend an element to the front of a vector.
--
-- >>> toList (cons 0 (fromList [1, 2, 3]))
-- [0, 1, 2, 3]
cons :: VectorElem a => a -> Vector a -> Vector a
cons x v = fromList (x : toList v)

-- | /O(n)/. Append an element to the end of a vector.
--
-- >>> toList (snoc (fromList [1, 2, 3]) 4)
-- [1, 2, 3, 4]
snoc :: VectorElem a => Vector a -> a -> Vector a
snoc v x = fromList (toList v P.++ [x])

-- | /O(n+m)/. Concatenate two vectors.
--
-- >>> toList (fromList [1, 2] ++ fromList [3, 4, 5])
-- [1, 2, 3, 4, 5]
(++) :: VectorElem a => Vector a -> Vector a -> Vector a
v1 ++ v2 = fromList (toList v1 P.++ toList v2)

-- | /O(n)/. Concatenate a list of vectors.
--
-- >>> toList (concat [fromList [1, 2], fromList [3], fromList [4, 5]])
-- [1, 2, 3, 4, 5]
concat :: VectorElem a => [Vector a] -> Vector a
concat vs = fromList (P.concatMap toList vs)

-- ============================================================
-- Element-wise Operations
-- ============================================================

-- | /O(n)/. Map a function over all elements of a vector.
--
-- >>> toList (map (*2) (fromList [1, 2, 3, 4]))
-- [2, 4, 6, 8]
--
-- ==== __Fusion__
--
-- In Numeric profile, @map f . map g@ fuses to @map (f . g)@.
map :: (VectorElem a, VectorElem b) => (a -> b) -> Vector a -> Vector b
map f v = generate (length v) (\i -> f (v ! i))

-- | /O(n)/. Map a function over all elements with their indices.
--
-- >>> toList (imap (\i x -> i + x) (fromList [10, 20, 30]))
-- [10, 21, 32]
imap :: (VectorElem a, VectorElem b) => (Int -> a -> b) -> Vector a -> Vector b
imap f v = generate (length v) (\i -> f i (v ! i))

-- | /O(min(n,m))/. Zip two vectors with a function.
--
-- >>> toList (zipWith (+) (fromList [1, 2, 3]) (fromList [10, 20, 30]))
-- [11, 22, 33]
--
-- >>> toList (zipWith (*) (fromList [1, 2]) (fromList [10, 20, 30]))
-- [10, 40]
zipWith :: (VectorElem a, VectorElem b, VectorElem c) => (a -> b -> c) -> Vector a -> Vector b -> Vector c
zipWith f va vb =
    let n = P.min (length va) (length vb)
    in generate n (\i -> f (va ! i) (vb ! i))

-- | /O(min(n,m,k))/. Zip three vectors with a function.
--
-- >>> toList (zipWith3 (\a b c -> a + b + c) (fromList [1,2,3]) (fromList [10,20,30]) (fromList [100,200,300]))
-- [111, 222, 333]
zipWith3 :: (VectorElem a, VectorElem b, VectorElem c, VectorElem d) => (a -> b -> c -> d) -> Vector a -> Vector b -> Vector c -> Vector d
zipWith3 f va vb vc =
    let n = P.minimum [length va, length vb, length vc]
    in generate n (\i -> f (va ! i) (vb ! i) (vc ! i))

-- | /O(min(n,m))/. Zip two vectors with a function that also takes the index.
--
-- >>> toList (izipWith (\i a b -> i * (a + b)) (fromList [1, 2]) (fromList [10, 20]))
-- [0, 22]
izipWith :: (VectorElem a, VectorElem b, VectorElem c) => (Int -> a -> b -> c) -> Vector a -> Vector b -> Vector c
izipWith f va vb =
    let n = P.min (length va) (length vb)
    in generate n (\i -> f i (va ! i) (vb ! i))

-- | /O(min(n,m,k))/. Zip three vectors with a function that also takes the index.
izipWith3 :: (VectorElem a, VectorElem b, VectorElem c, VectorElem d) => (Int -> a -> b -> c -> d) -> Vector a -> Vector b -> Vector c -> Vector d
izipWith3 f va vb vc =
    let n = P.minimum [length va, length vb, length vc]
    in generate n (\i -> f i (va ! i) (vb ! i) (vc ! i))

-- ============================================================
-- Folds
-- ============================================================

-- | /O(n)/. Left-associative fold.
--
-- >>> foldl (+) 0 (fromList [1, 2, 3, 4])
-- 10
--
-- >>> foldl (flip (:)) [] (fromList [1, 2, 3])
-- [3, 2, 1]
foldl :: VectorElem a => (b -> a -> b) -> b -> Vector a -> b
foldl f z v = go 0 z
  where
    n = length v
    go !i !acc
        | i >= n    = acc
        | otherwise = go (i + 1) (f acc (v ! i))

-- | /O(n)/. Strict left-associative fold.
--
-- Prefer 'foldl'' over 'foldl' to avoid space leaks.
--
-- >>> foldl' (+) 0 (fromList [1, 2, 3, 4])
-- 10
foldl' :: VectorElem a => (b -> a -> b) -> b -> Vector a -> b
foldl' = foldl  -- Already strict due to bang patterns

-- | /O(n)/. Left fold without a starting value.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> foldl1 (+) (fromList [1, 2, 3, 4])
-- 10
foldl1 :: VectorElem a => (a -> a -> a) -> Vector a -> a
foldl1 f v = foldl f (head v) (tail v)

-- | /O(n)/. Strict left fold without a starting value.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
foldl1' :: VectorElem a => (a -> a -> a) -> Vector a -> a
foldl1' = foldl1

-- | /O(n)/. Right-associative fold.
--
-- >>> foldr (:) [] (fromList [1, 2, 3])
-- [1, 2, 3]
foldr :: VectorElem a => (a -> b -> b) -> b -> Vector a -> b
foldr f z v = go (length v - 1)
  where
    go i
        | i < 0     = z
        | otherwise = f (v ! i) (go (i - 1))

-- | /O(n)/. Right fold without a starting value.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> foldr1 (+) (fromList [1, 2, 3, 4])
-- 10
foldr1 :: VectorElem a => (a -> a -> a) -> Vector a -> a
foldr1 f v = foldr f (last v) (init v)

-- | /O(n)/. Left fold with index.
--
-- >>> ifoldl (\acc i x -> acc + i * x) 0 (fromList [10, 20, 30])
-- 80
ifoldl :: VectorElem a => (b -> Int -> a -> b) -> b -> Vector a -> b
ifoldl f z v = go 0 z
  where
    n = length v
    go !i !acc
        | i >= n    = acc
        | otherwise = go (i + 1) (f acc i (v ! i))

-- | /O(n)/. Strict left fold with index.
ifoldl' :: VectorElem a => (b -> Int -> a -> b) -> b -> Vector a -> b
ifoldl' = ifoldl

-- | /O(n)/. Right fold with index.
ifoldr :: VectorElem a => (Int -> a -> b -> b) -> b -> Vector a -> b
ifoldr f z v = go 0
  where
    n = length v
    go i
        | i >= n    = z
        | otherwise = f i (v ! i) (go (i + 1))

-- ============================================================
-- Specialized Folds
-- ============================================================

-- | /O(n)/. Sum of all elements.
--
-- >>> sum (fromList [1, 2, 3, 4, 5])
-- 15
--
-- ==== __Fusion__
--
-- In Numeric profile, @sum . map f@ fuses to a single traversal.
sum :: (Num a, VectorElem a) => Vector a -> a
sum = foldl' (+) 0

-- | /O(n)/. Product of all elements.
--
-- >>> product (fromList [1, 2, 3, 4, 5])
-- 120
product :: (Num a, VectorElem a) => Vector a -> a
product = foldl' (*) 1

-- | /O(n)/. The maximum element of a vector.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> maximum (fromList [3, 1, 4, 1, 5, 9])
-- 9
maximum :: (Ord a, VectorElem a) => Vector a -> a
maximum = foldl1' P.max

-- | /O(n)/. The minimum element of a vector.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> minimum (fromList [3, 1, 4, 1, 5, 9])
-- 1
minimum :: (Ord a, VectorElem a) => Vector a -> a
minimum = foldl1' P.min

-- | /O(n)/. The maximum element using a custom comparison function.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> maximumBy (comparing abs) (fromList [-5, 3, -2, 4])
-- -5
maximumBy :: VectorElem a => (a -> a -> Ordering) -> Vector a -> a
maximumBy cmp = foldl1' (\a b -> if cmp a b == GT then a else b)

-- | /O(n)/. The minimum element using a custom comparison function.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> minimumBy (comparing abs) (fromList [-5, 3, -2, 4])
-- -2
minimumBy :: VectorElem a => (a -> a -> Ordering) -> Vector a -> a
minimumBy cmp = foldl1' (\a b -> if cmp a b == LT then a else b)

-- | /O(n)/. Test whether all elements satisfy a predicate.
--
-- >>> all even (fromList [2, 4, 6, 8])
-- True
-- >>> all even (fromList [2, 3, 6, 8])
-- False
all :: VectorElem a => (a -> Bool) -> Vector a -> Bool
all p = foldl' (\acc x -> acc P.&& p x) True

-- | /O(n)/. Test whether any element satisfies a predicate.
--
-- >>> any even (fromList [1, 3, 5, 8])
-- True
-- >>> any even (fromList [1, 3, 5, 7])
-- False
any :: VectorElem a => (a -> Bool) -> Vector a -> Bool
any p = foldl' (\acc x -> acc P.|| p x) False

-- ============================================================
-- Scans
-- ============================================================

-- | /O(n²)/. Left-to-right exclusive prefix scan.
--
-- @prescanl f z [x1, x2, ...] = [z, f z x1, f (f z x1) x2, ...]@
--
-- >>> toList (prescanl (+) 0 (fromList [1, 2, 3, 4]))
-- [0, 1, 3, 6]
prescanl :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
prescanl f z v = generate (length v) (\i ->
    foldl' f z (take i v))

-- | /O(n²)/. Strict left-to-right exclusive prefix scan.
prescanl' :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
prescanl' = prescanl

-- | /O(n²)/. Left-to-right inclusive prefix scan.
--
-- @postscanl f z [x1, x2, ...] = [f z x1, f (f z x1) x2, ...]@
--
-- >>> toList (postscanl (+) 0 (fromList [1, 2, 3, 4]))
-- [1, 3, 6, 10]
postscanl :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
postscanl f z v = generate (length v) (\i ->
    foldl' f z (take (i + 1) v))

-- | /O(n²)/. Strict left-to-right inclusive prefix scan.
postscanl' :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
postscanl' = postscanl

-- | /O(n²)/. Left scan (similar to 'prescanl').
--
-- >>> toList (scanl (+) 0 (fromList [1, 2, 3, 4]))
-- [0, 1, 3, 6]
scanl :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
scanl = prescanl

-- | /O(n²)/. Strict left scan.
scanl' :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
scanl' = prescanl'

-- | /O(n²)/. Left scan without a starting value.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
--
-- >>> toList (scanl1 (+) (fromList [1, 2, 3, 4]))
-- [1, 3, 6, 10]
scanl1 :: VectorElem a => (a -> a -> a) -> Vector a -> Vector a
scanl1 f v = postscanl f (head v) (tail v)

-- | /O(n²)/. Strict left scan without a starting value.
--
-- __Warning__: Partial function. Throws an error on empty vectors.
scanl1' :: VectorElem a => (a -> a -> a) -> Vector a -> Vector a
scanl1' = scanl1

-- ============================================================
-- Search
-- ============================================================

-- | /O(n)/. Test whether an element is in the vector.
--
-- >>> elem 3 (fromList [1, 2, 3, 4, 5])
-- True
-- >>> elem 6 (fromList [1, 2, 3, 4, 5])
-- False
elem :: (Eq a, VectorElem a) => a -> Vector a -> Bool
elem x = any (== x)

-- | /O(n)/. Test whether an element is not in the vector.
--
-- >>> notElem 6 (fromList [1, 2, 3, 4, 5])
-- True
notElem :: (Eq a, VectorElem a) => a -> Vector a -> Bool
notElem x = P.not . elem x

-- | /O(n)/. Find the first element satisfying a predicate.
--
-- >>> find even (fromList [1, 3, 4, 5, 6])
-- Just 4
-- >>> find (> 10) (fromList [1, 3, 4, 5, 6])
-- Nothing
find :: VectorElem a => (a -> Bool) -> Vector a -> Maybe a
find p v = case findIndex p v of
    Nothing -> Nothing
    Just i  -> Just (v ! i)

-- | /O(n)/. Find the index of the first element satisfying a predicate.
--
-- >>> findIndex even (fromList [1, 3, 4, 5, 6])
-- Just 2
-- >>> findIndex (> 10) (fromList [1, 3, 4, 5, 6])
-- Nothing
findIndex :: VectorElem a => (a -> Bool) -> Vector a -> Maybe Int
findIndex p v = go 0
  where
    n = length v
    go i
        | i >= n    = Nothing
        | p (v ! i) = Just i
        | otherwise = go (i + 1)

-- | /O(n)/. Find the index of the first occurrence of an element.
--
-- >>> elemIndex 30 (fromList [10, 20, 30, 40, 30])
-- Just 2
-- >>> elemIndex 50 (fromList [10, 20, 30, 40])
-- Nothing
elemIndex :: (Eq a, VectorElem a) => a -> Vector a -> Maybe Int
elemIndex x = findIndex (== x)

-- | /O(n)/. Find all indices where an element occurs.
--
-- >>> toList (elemIndices 30 (fromList [10, 30, 20, 30, 40]))
-- [1, 3]
elemIndices :: (Eq a, VectorElem a) => a -> Vector a -> Vector Int
elemIndices x v = fromList [i | i <- [0..length v - 1], v ! i == x]

-- ============================================================
-- Modification
-- ============================================================

-- | /O(n)/. Update the element at a given index.
--
-- Returns the original vector unchanged if the index is out of bounds.
--
-- >>> toList (update (fromList [1, 2, 3]) 1 10)
-- [1, 10, 3]
--
-- >>> toList (update (fromList [1, 2, 3]) 5 10)
-- [1, 2, 3]
update :: VectorElem a => Vector a -> Int -> a -> Vector a
update v i x
    | i < 0 || i >= length v = v
    | otherwise = generate (length v) (\j -> if j == i then x else v ! j)

-- | /O(n)/. Keep only elements that satisfy a predicate.
--
-- >>> toList (filter (> 2) (fromList [1, 2, 3, 4, 5]))
-- [3, 4, 5]
--
-- >>> toList (filter even (fromList [1, 2, 3, 4, 5, 6]))
-- [2, 4, 6]
filter :: VectorElem a => (a -> Bool) -> Vector a -> Vector a
filter p v = fromList [v ! i | i <- [0..length v - 1], p (v ! i)]

-- ============================================================
-- Numeric Operations
-- ============================================================

-- | /O(n)/. Compute the dot product (inner product) of two vectors.
--
-- \[ \text{dot}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} x_i \cdot y_i \]
--
-- >>> dot (fromList [1, 2, 3]) (fromList [4, 5, 6])
-- 32.0
--
-- ==== __SIMD__
--
-- For 'Float' and 'Double' vectors, this uses SIMD instructions
-- (AVX-256) when available, processing 8 floats or 4 doubles per cycle.
dot :: VectorElem a => Vector a -> Vector a -> a
dot v1 v2 = unsafePerformIO $ vectorDot v1 v2
{-# NOINLINE dot #-}

-- | /O(n)/. Compute the Euclidean norm (L2 norm) of a vector.
--
-- \[ \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} \]
--
-- >>> norm (fromList [3.0, 4.0])
-- 5.0
norm :: (Floating a, VectorElem a) => Vector a -> a
norm v = P.sqrt (dot v v)

-- | /O(n)/. Normalize a vector to unit length.
--
-- >>> let v = normalize (fromList [3.0, 4.0])
-- >>> norm v
-- 1.0
--
-- __Warning__: Divides by the norm, which may cause issues for
-- zero vectors.
normalize :: (Floating a, VectorElem a) => Vector a -> Vector a
normalize v = scale (1 / norm v) v

-- | /O(n)/. Element-wise addition of two vectors.
--
-- >>> toList (add (fromList [1, 2, 3]) (fromList [10, 20, 30]))
-- [11, 22, 33]
add :: (Num a, VectorElem a) => Vector a -> Vector a -> Vector a
add = zipWith (+)

-- | /O(n)/. Element-wise subtraction of two vectors.
--
-- >>> toList (sub (fromList [10, 20, 30]) (fromList [1, 2, 3]))
-- [9, 18, 27]
sub :: (Num a, VectorElem a) => Vector a -> Vector a -> Vector a
sub = zipWith (-)

-- | /O(n)/. Element-wise multiplication (Hadamard product).
--
-- >>> toList (mul (fromList [1, 2, 3]) (fromList [4, 5, 6]))
-- [4, 10, 18]
mul :: (Num a, VectorElem a) => Vector a -> Vector a -> Vector a
mul = zipWith (*)

-- | /O(n)/. Element-wise division of two vectors.
--
-- >>> toList (div (fromList [10.0, 20.0, 30.0]) (fromList [2.0, 4.0, 5.0]))
-- [5.0, 5.0, 6.0]
div :: (Fractional a, VectorElem a) => Vector a -> Vector a -> Vector a
div = zipWith (/)

-- | /O(n)/. Scale all elements by a constant factor.
--
-- >>> toList (scale 3 (fromList [1, 2, 3]))
-- [3, 6, 9]
scale :: (Num a, VectorElem a) => a -> Vector a -> Vector a
scale k = map (* k)

-- | /O(n)/. Negate all elements of a vector.
--
-- >>> toList (negate (fromList [1, -2, 3]))
-- [-1, 2, -3]
negate :: (Num a, VectorElem a) => Vector a -> Vector a
negate = map P.negate

-- | /O(n)/. Absolute value of all elements.
--
-- >>> toList (abs (fromList [-3, 2, -1, 4]))
-- [3, 2, 1, 4]
abs :: (Num a, VectorElem a) => Vector a -> Vector a
abs = map P.abs

-- ============================================================
-- Sorting
-- ============================================================

-- | /O(n log n)/. Sort the elements in ascending order.
--
-- >>> toList (sort (fromList [3, 1, 4, 1, 5, 9, 2, 6]))
-- [1, 1, 2, 3, 4, 5, 6, 9]
sort :: (Ord a, VectorElem a) => Vector a -> Vector a
sort v = fromList (P.sort (toList v))

-- | /O(n log n)/. Sort using a custom comparison function.
--
-- >>> toList (sortBy (comparing Down) (fromList [1, 3, 2, 5, 4]))
-- [5, 4, 3, 2, 1]
sortBy :: VectorElem a => (a -> a -> Ordering) -> Vector a -> Vector a
sortBy cmp v = fromList (P.sortBy cmp (toList v))

-- | /O(n)/. Pair each element with its index.
--
-- >>> toList (indexed (fromList ['a', 'b', 'c']))
-- [(0, 'a'), (1, 'b'), (2, 'c')]
indexed :: (VectorElem a, VectorElem (Int, a)) => Vector a -> Vector (Int, a)
indexed v = generate (length v) (\i -> (i, v ! i))

-- ============================================================
-- Conversion
-- ============================================================

-- | /O(n)/. Convert a vector to a list.
--
-- >>> toList (fromList [1, 2, 3, 4, 5])
-- [1, 2, 3, 4, 5]
toList :: VectorElem a => Vector a -> [a]
toList v = [v ! i | i <- [0..length v - 1]]

-- | /O(1)/. Convert between vector and unboxed vector representations.
--
-- This is a zero-copy operation that reinterprets the underlying buffer.
convert :: Vector a -> UVector a
convert (Vector fp off len) = UVector fp off len
