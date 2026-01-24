-- |
-- Module      : BHC.Data.IntSet
-- Description : Efficient sets of Int values
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- An efficient implementation of sets of integers using Patricia tries.
--
-- = Performance
--
-- Most operations are /O(min(n, W))/ where /W/ is the number of bits
-- in an 'Int' (typically 64). This uses bitmap compression at the leaves
-- for additional space efficiency.
--
-- | Operation     | Time Complexity |
-- |---------------|-----------------|
-- | member        | O(min(n, W))    |
-- | insert        | O(min(n, W))    |
-- | delete        | O(min(n, W))    |
-- | union         | O(n + m)        |
-- | intersection  | O(n + m)        |
--
-- = Usage
--
-- This module is designed to be imported qualified:
--
-- @
-- import qualified BHC.Data.IntSet as IS
--
-- primes :: IS.IntSet
-- primes = IS.fromList [2, 3, 5, 7, 11, 13]
--
-- isPrime :: Int -> Bool
-- isPrime = flip IS.member primes
-- @
--
-- = Comparison with Data.Set
--
-- Use 'IntSet' when storing 'Int' values. It is significantly faster
-- than @Set Int@ and uses less memory due to bitmap compression.

{-# LANGUAGE BangPatterns #-}

module BHC.Data.IntSet (
    -- * Set type
    IntSet,

    -- * Construction
    empty,
    singleton,
    fromList,
    fromAscList,
    fromDistinctAscList,

    -- * Insertion
    insert,

    -- * Deletion
    delete,

    -- * Query
    member,
    notMember,
    lookupLT,
    lookupGT,
    lookupLE,
    lookupGE,
    null,
    size,
    isSubsetOf,
    isProperSubsetOf,
    disjoint,

    -- * Combine
    union,
    unions,
    difference,
    (\\),
    intersection,

    -- * Filter
    filter,
    partition,
    split,
    splitMember,
    splitRoot,

    -- * Map
    map,

    -- * Folds
    foldr,
    foldl,
    foldr',
    foldl',

    -- * Min/Max
    findMin,
    findMax,
    lookupMin,
    lookupMax,
    deleteMin,
    deleteMax,
    deleteFindMin,
    deleteFindMax,
    minView,
    maxView,

    -- * Conversion
    elems,
    toList,
    toAscList,
    toDescList,
) where

import Prelude hiding (lookup, map, filter, foldr, foldl, null)
import qualified Prelude as P
import Data.Bits
import Data.Maybe (fromMaybe)

-- ============================================================
-- Type
-- ============================================================

-- | A set of 'Int' values.
--
-- Uses Patricia tries for O(min(n, W)) operations where W is the
-- number of bits in an 'Int' (typically 64).
data IntSet
    = Nil
    | Tip {-# UNPACK #-} !Key {-# UNPACK #-} !BitMap
    | Bin {-# UNPACK #-} !Prefix
          {-# UNPACK #-} !Mask
          !IntSet
          !IntSet
    deriving (Eq, Ord)

type Key = Int
type Prefix = Int
type Mask = Int
type BitMap = Word

instance Show IntSet where
    showsPrec d s = showParen (d > 10) $
        showString "fromList " . shows (toList s)

instance Semigroup IntSet where
    (<>) = union

instance Monoid IntSet where
    mempty = empty
    mappend = (<>)

-- ============================================================
-- Constants
-- ============================================================

-- | Number of bits in the bitmap (word size).
bitsPerWord :: Int
bitsPerWord = finiteBitSize (0 :: Word)

-- | Mask for extracting suffix (bits within a tip).
suffixMask :: Int
suffixMask = bitsPerWord - 1

-- | Mask for extracting prefix (bits above the tip).
prefixMask :: Int
prefixMask = complement suffixMask

-- ============================================================
-- Construction
-- ============================================================

-- | The empty set.
empty :: IntSet
empty = Nil

-- | A set with a single element.
singleton :: Key -> IntSet
singleton k = Tip (k .&. prefixMask) (bitmapOf k)

-- | Build a set from a list.
fromList :: [Key] -> IntSet
fromList = P.foldl (flip insert) empty

-- | Build a set from an ascending list.
fromAscList :: [Key] -> IntSet
fromAscList = fromList

-- | Build a set from an ascending list of distinct elements.
fromDistinctAscList :: [Key] -> IntSet
fromDistinctAscList = fromList

-- ============================================================
-- Insertion
-- ============================================================

-- | Insert an element.
insert :: Key -> IntSet -> IntSet
insert !k t = case t of
    Nil -> singleton k
    Tip p bm
        | p == prefixOf k -> Tip p (bm .|. bitmapOf k)
        | otherwise       -> link (prefixOf k) (singleton k) p t
    Bin p m l r
        | nomatch k p m -> link (prefixOf k) (singleton k) p t
        | zero k m      -> Bin p m (insert k l) r
        | otherwise     -> Bin p m l (insert k r)

-- ============================================================
-- Deletion
-- ============================================================

-- | Delete an element.
delete :: Key -> IntSet -> IntSet
delete !k t = case t of
    Nil -> Nil
    Tip p bm
        | p == prefixOf k -> tip p (bm .&. complement (bitmapOf k))
        | otherwise       -> t
    Bin p m l r
        | nomatch k p m -> t
        | zero k m      -> bin p m (delete k l) r
        | otherwise     -> bin p m l (delete k r)

-- ============================================================
-- Query
-- ============================================================

-- | Is the element in the set?
member :: Key -> IntSet -> Bool
member !k t = case t of
    Nil -> False
    Tip p bm -> p == prefixOf k && (bm .&. bitmapOf k) /= 0
    Bin p m l r
        | nomatch k p m -> False
        | zero k m      -> member k l
        | otherwise     -> member k r

-- | Is the element not in the set?
notMember :: Key -> IntSet -> Bool
notMember k = not . member k

-- | Find largest element smaller than the given one.
lookupLT :: Key -> IntSet -> Maybe Key
lookupLT k t = goNothing t
  where
    goNothing Nil = Nothing
    goNothing (Tip p bm)
        | p >= prefixOf k = Nothing
        | otherwise       = Just (highestBitInBitMap p bm)
    goNothing (Bin p m l r)
        | nomatch k p m   = if k > p then goNothing r else Nothing
        | zero k m        = goNothing l
        | otherwise       = case goNothing r of
            Nothing -> goNothing l
            just    -> just
    highestBitInBitMap p bm = p + (bitsPerWord - 1 - countLeadingZeros bm)

-- | Find smallest element larger than the given one.
lookupGT :: Key -> IntSet -> Maybe Key
lookupGT k t = goNothing t
  where
    goNothing Nil = Nothing
    goNothing (Tip p bm)
        | p <= prefixOf k = Nothing
        | otherwise       = Just (lowestBitInBitMap p bm)
    goNothing (Bin p m l r)
        | nomatch k p m   = if k < p then goNothing l else Nothing
        | zero k m        = case goNothing l of
            Nothing -> goNothing r
            just    -> just
        | otherwise       = goNothing r
    lowestBitInBitMap p bm = p + countTrailingZeros bm

-- | Find largest element smaller than or equal to the given one.
lookupLE :: Key -> IntSet -> Maybe Key
lookupLE k s = if member k s then Just k else lookupLT k s

-- | Find smallest element larger than or equal to the given one.
lookupGE :: Key -> IntSet -> Maybe Key
lookupGE k s = if member k s then Just k else lookupGT k s

-- | Is the set empty?
null :: IntSet -> Bool
null Nil = True
null _   = False

-- | Number of elements in the set.
size :: IntSet -> Int
size = go 0
  where
    go !acc Nil           = acc
    go !acc (Tip _ bm)    = acc + popCount bm
    go !acc (Bin _ _ l r) = go (go acc l) r

-- | Is the first set a subset of the second?
isSubsetOf :: IntSet -> IntSet -> Bool
isSubsetOf t1 t2 = case (t1, t2) of
    (Nil, _)   -> True
    (_, Nil)   -> False
    (Tip p1 bm1, Tip p2 bm2) -> p1 == p2 && (bm1 .&. bm2) == bm1
    (Tip p1 _, Bin p2 m2 l r)
        | nomatch p1 p2 m2 -> False
        | zero p1 m2       -> isSubsetOf t1 l
        | otherwise        -> isSubsetOf t1 r
    (Bin _ _ _ _, Tip _ _) -> False
    (Bin p1 m1 l1 r1, Bin p2 m2 l2 r2)
        | shorter m1 m2 -> False
        | shorter m2 m1 -> match p1 p2 m2 &&
                          (if zero p1 m2 then isSubsetOf t1 l2 else isSubsetOf t1 r2)
        | otherwise     -> p1 == p2 && isSubsetOf l1 l2 && isSubsetOf r1 r2

-- | Is the first set a proper subset of the second?
isProperSubsetOf :: IntSet -> IntSet -> Bool
isProperSubsetOf t1 t2 = size t1 < size t2 && isSubsetOf t1 t2

-- | Are the two sets disjoint?
disjoint :: IntSet -> IntSet -> Bool
disjoint t1 t2 = null (intersection t1 t2)

-- ============================================================
-- Combine
-- ============================================================

-- | Union of two sets.
union :: IntSet -> IntSet -> IntSet
union t1 t2 = case (t1, t2) of
    (Nil, t) -> t
    (t, Nil) -> t
    (Tip p1 bm1, Tip p2 bm2)
        | p1 == p2  -> Tip p1 (bm1 .|. bm2)
        | otherwise -> link p1 t1 p2 t2
    (Tip p1 _, Bin p2 m2 l r)
        | nomatch p1 p2 m2 -> link p1 t1 p2 t2
        | zero p1 m2       -> Bin p2 m2 (union t1 l) r
        | otherwise        -> Bin p2 m2 l (union t1 r)
    (Bin p1 m1 l r, Tip p2 _)
        | nomatch p2 p1 m1 -> link p1 t1 p2 t2
        | zero p2 m1       -> Bin p1 m1 (union l t2) r
        | otherwise        -> Bin p1 m1 l (union r t2)
    (Bin p1 m1 l1 r1, Bin p2 m2 l2 r2)
        | shorter m1 m2 -> union1
        | shorter m2 m1 -> union2
        | p1 == p2      -> Bin p1 m1 (union l1 l2) (union r1 r2)
        | otherwise     -> link p1 t1 p2 t2
      where
        union1
            | nomatch p2 p1 m1 = link p1 t1 p2 t2
            | zero p2 m1       = Bin p1 m1 (union l1 t2) r1
            | otherwise        = Bin p1 m1 l1 (union r1 t2)
        union2
            | nomatch p1 p2 m2 = link p1 t1 p2 t2
            | zero p1 m2       = Bin p2 m2 (union t1 l2) r2
            | otherwise        = Bin p2 m2 l2 (union t1 r2)

-- | Union of a list of sets.
unions :: [IntSet] -> IntSet
unions = P.foldl union empty

-- | Difference of two sets.
difference :: IntSet -> IntSet -> IntSet
difference t1 t2 = case (t1, t2) of
    (Nil, _)   -> Nil
    (t, Nil)   -> t
    (Tip p1 bm1, Tip p2 bm2)
        | p1 == p2  -> tip p1 (bm1 .&. complement bm2)
        | otherwise -> t1
    (Tip p1 _, Bin p2 m2 l r)
        | nomatch p1 p2 m2 -> t1
        | zero p1 m2       -> difference t1 l
        | otherwise        -> difference t1 r
    (Bin p1 m1 l r, Tip p2 _)
        | nomatch p2 p1 m1 -> t1
        | zero p2 m1       -> bin p1 m1 (difference l t2) r
        | otherwise        -> bin p1 m1 l (difference r t2)
    (Bin p1 m1 l1 r1, Bin p2 m2 l2 r2)
        | shorter m1 m2 -> diff1
        | shorter m2 m1 -> diff2
        | p1 == p2      -> bin p1 m1 (difference l1 l2) (difference r1 r2)
        | otherwise     -> t1
      where
        diff1
            | nomatch p2 p1 m1 = t1
            | zero p2 m1       = bin p1 m1 (difference l1 t2) r1
            | otherwise        = bin p1 m1 l1 (difference r1 t2)
        diff2
            | nomatch p1 p2 m2 = t1
            | zero p1 m2       = difference t1 l2
            | otherwise        -> difference t1 r2

-- | Difference operator.
(\\) :: IntSet -> IntSet -> IntSet
(\\) = difference
infixl 9 \\

-- | Intersection of two sets.
intersection :: IntSet -> IntSet -> IntSet
intersection t1 t2 = case (t1, t2) of
    (Nil, _)   -> Nil
    (_, Nil)   -> Nil
    (Tip p1 bm1, Tip p2 bm2)
        | p1 == p2  -> tip p1 (bm1 .&. bm2)
        | otherwise -> Nil
    (Tip p1 _, Bin p2 m2 l r)
        | nomatch p1 p2 m2 -> Nil
        | zero p1 m2       -> intersection t1 l
        | otherwise        -> intersection t1 r
    (Bin p1 m1 l r, Tip p2 _)
        | nomatch p2 p1 m1 -> Nil
        | zero p2 m1       -> intersection l t2
        | otherwise        -> intersection r t2
    (Bin p1 m1 l1 r1, Bin p2 m2 l2 r2)
        | shorter m1 m2 -> inter1
        | shorter m2 m1 -> inter2
        | p1 == p2      -> bin p1 m1 (intersection l1 l2) (intersection r1 r2)
        | otherwise     -> Nil
      where
        inter1
            | nomatch p2 p1 m1 = Nil
            | zero p2 m1       = intersection l1 t2
            | otherwise        = intersection r1 t2
        inter2
            | nomatch p1 p2 m2 = Nil
            | zero p1 m2       = intersection t1 l2
            | otherwise        = intersection t1 r2

-- ============================================================
-- Filter
-- ============================================================

-- | Filter elements satisfying a predicate.
filter :: (Key -> Bool) -> IntSet -> IntSet
filter p = go
  where
    go Nil = Nil
    go (Tip prefix bm) = tip prefix (foldlBits prefix 0 addIfMatch bm)
      where
        addIfMatch acc k = if p k then acc .|. bitmapOf k else acc
    go (Bin prefix m l r) = bin prefix m (go l) (go r)

-- | Partition by a predicate.
partition :: (Key -> Bool) -> IntSet -> (IntSet, IntSet)
partition p s = (filter p s, filter (not . p) s)

-- | Split at a value.
split :: Key -> IntSet -> (IntSet, IntSet)
split k s = case splitMember k s of
    (lt, _, gt) -> (lt, gt)

-- | Split and report membership.
splitMember :: Key -> IntSet -> (IntSet, Bool, IntSet)
splitMember k t = case t of
    Nil -> (Nil, False, Nil)
    Tip p bm
        | p > prefixOf k  -> (Nil, False, t)
        | p < prefixOf k  -> (t, False, Nil)
        | otherwise ->
            let ltBm = bm .&. (bitmapOf k - 1)
                gtBm = bm .&. complement (bitmapOf k .|. (bitmapOf k - 1))
                found = (bm .&. bitmapOf k) /= 0
            in (tip p ltBm, found, tip p gtBm)
    Bin p m l r
        | nomatch k p m -> if k > p then (t, False, Nil) else (Nil, False, t)
        | zero k m      -> let (lt, found, gt) = splitMember k l
                           in (lt, found, bin p m gt r)
        | otherwise     -> let (lt, found, gt) = splitMember k r
                           in (bin p m l lt, found, gt)

-- | Decompose into pieces based on structure.
splitRoot :: IntSet -> [IntSet]
splitRoot Nil           = []
splitRoot (Tip p bm)    = [Tip p bm]
splitRoot (Bin _ _ l r) = [l, r]

-- ============================================================
-- Map
-- ============================================================

-- | Map a function over elements.
--
-- Note: If the function is not injective, the resulting set may be smaller.
map :: (Key -> Key) -> IntSet -> IntSet
map f = fromList . P.map f . toList

-- ============================================================
-- Folds
-- ============================================================

-- | Lazy right fold.
foldr :: (Key -> b -> b) -> b -> IntSet -> b
foldr f z = go
  where
    go Nil           = z
    go (Tip p bm)    = foldrBits p f z bm
    go (Bin _ _ l r) = go l `seq` go r `seq` go l `f'` go r
      where f' a b = P.foldr f b (toAscListAux a [])
            toAscListAux Nil xs = xs
            toAscListAux (Tip p' bm) xs = foldrBits p' (:) xs bm
            toAscListAux (Bin _ _ l' r') xs = toAscListAux l' (toAscListAux r' xs)

-- | Lazy left fold.
foldl :: (b -> Key -> b) -> b -> IntSet -> b
foldl f z = go z
  where
    go !acc Nil           = acc
    go !acc (Tip p bm)    = foldlBits p acc f bm
    go !acc (Bin _ _ l r) = go (go acc l) r

-- | Strict right fold.
foldr' :: (Key -> b -> b) -> b -> IntSet -> b
foldr' f !z = go
  where
    go Nil           = z
    go (Tip p bm)    = foldrBits p f z bm
    go (Bin _ _ l r) = let !z' = go r in go' z' l
    go' !acc Nil           = acc
    go' !acc (Tip p bm)    = foldrBits p f acc bm
    go' !acc (Bin _ _ l r) = let !acc' = go' acc r in go' acc' l

-- | Strict left fold.
foldl' :: (b -> Key -> b) -> b -> IntSet -> b
foldl' f !z = go z
  where
    go !acc Nil           = acc
    go !acc (Tip p bm)    = foldlBits p acc f bm
    go !acc (Bin _ _ l r) = go (go acc l) r

-- ============================================================
-- Min/Max
-- ============================================================

-- | Find minimum element (partial).
findMin :: IntSet -> Key
findMin = fromMaybe (P.error "IntSet.findMin: empty set") . lookupMin

-- | Find maximum element (partial).
findMax :: IntSet -> Key
findMax = fromMaybe (P.error "IntSet.findMax: empty set") . lookupMax

-- | Lookup minimum element.
lookupMin :: IntSet -> Maybe Key
lookupMin Nil           = Nothing
lookupMin (Tip p bm)    = Just (p + countTrailingZeros bm)
lookupMin (Bin _ _ l _) = lookupMin l

-- | Lookup maximum element.
lookupMax :: IntSet -> Maybe Key
lookupMax Nil           = Nothing
lookupMax (Tip p bm)    = Just (p + bitsPerWord - 1 - countLeadingZeros bm)
lookupMax (Bin _ _ _ r) = lookupMax r

-- | Delete minimum element.
deleteMin :: IntSet -> IntSet
deleteMin = maybe empty snd . minView

-- | Delete maximum element.
deleteMax :: IntSet -> IntSet
deleteMax = maybe empty snd . maxView

-- | Delete and return minimum.
deleteFindMin :: IntSet -> (Key, IntSet)
deleteFindMin s = case minView s of
    Nothing      -> P.error "IntSet.deleteFindMin: empty set"
    Just (k, s') -> (k, s')

-- | Delete and return maximum.
deleteFindMax :: IntSet -> (Key, IntSet)
deleteFindMax s = case maxView s of
    Nothing      -> P.error "IntSet.deleteFindMax: empty set"
    Just (k, s') -> (k, s')

-- | View with minimum removed.
minView :: IntSet -> Maybe (Key, IntSet)
minView Nil           = Nothing
minView (Tip p bm)    =
    let i = countTrailingZeros bm
        k = p + i
        bm' = bm .&. complement (bit i)
    in Just (k, tip p bm')
minView (Bin p m l r) = case minView l of
    Nothing      -> minView r
    Just (k, l') -> Just (k, bin p m l' r)

-- | View with maximum removed.
maxView :: IntSet -> Maybe (Key, IntSet)
maxView Nil           = Nothing
maxView (Tip p bm)    =
    let i = bitsPerWord - 1 - countLeadingZeros bm
        k = p + i
        bm' = bm .&. complement (bit i)
    in Just (k, tip p bm')
maxView (Bin p m l r) = case maxView r of
    Nothing      -> maxView l
    Just (k, r') -> Just (k, bin p m l r')

-- ============================================================
-- Conversion
-- ============================================================

-- | All elements in ascending order.
elems :: IntSet -> [Key]
elems = toAscList

-- | Convert to a list.
toList :: IntSet -> [Key]
toList = toAscList

-- | Convert to an ascending list.
toAscList :: IntSet -> [Key]
toAscList = foldr (:) []

-- | Convert to a descending list.
toDescList :: IntSet -> [Key]
toDescList = foldl (flip (:)) []

-- ============================================================
-- Internal Helpers
-- ============================================================

-- | Link two disjoint trees.
link :: Prefix -> IntSet -> Prefix -> IntSet -> IntSet
link p1 t1 p2 t2
    | zero p1 m = Bin p m t1 t2
    | otherwise = Bin p m t2 t1
  where
    m = branchMask p1 p2
    p = mask p1 m

-- | Smart constructor for Tip.
tip :: Prefix -> BitMap -> IntSet
tip _ 0  = Nil
tip p bm = Tip p bm

-- | Smart constructor for Bin.
bin :: Prefix -> Mask -> IntSet -> IntSet -> IntSet
bin _ _ l Nil = l
bin _ _ Nil r = r
bin p m l r   = Bin p m l r

-- | Check if bit is zero.
zero :: Key -> Mask -> Bool
zero k m = (k .&. m) == 0

-- | Check for match with prefix.
nomatch :: Key -> Prefix -> Mask -> Bool
nomatch k p m = mask k m /= p

-- | Check for match.
match :: Key -> Prefix -> Mask -> Bool
match k p m = mask k m == p

-- | Get prefix.
mask :: Key -> Mask -> Prefix
mask k m = k .&. complement (m - 1) `xor` m

-- | Get the prefix of a key (upper bits).
prefixOf :: Key -> Prefix
prefixOf k = k .&. prefixMask

-- | Get the bitmap for a key.
bitmapOf :: Key -> BitMap
bitmapOf k = bit (k .&. suffixMask)

-- | Is the first mask shorter (higher bit)?
shorter :: Mask -> Mask -> Bool
shorter m1 m2 = m1 > m2

-- | Find branching bit.
branchMask :: Prefix -> Prefix -> Mask
branchMask p1 p2 = highestBitMask (p1 `xor` p2)

-- | Get highest set bit as mask.
highestBitMask :: Int -> Int
highestBitMask x = x' `xor` (x' `shiftR` 1)
  where
    x' = P.foldr (.|.) x [shiftR x i | i <- [1, 2, 4, 8, 16, 32]]

-- | Fold right over bits in a bitmap.
foldrBits :: Prefix -> (Key -> b -> b) -> b -> BitMap -> b
foldrBits prefix f z bm = go (bitsPerWord - 1) z
  where
    go !i !acc
        | i < 0     = acc
        | testBit bm i = go (i - 1) (f (prefix + i) acc)
        | otherwise = go (i - 1) acc

-- | Fold left over bits in a bitmap.
foldlBits :: Prefix -> b -> (b -> Key -> b) -> BitMap -> b
foldlBits prefix z f bm = go 0 z
  where
    go !i !acc
        | i >= bitsPerWord = acc
        | testBit bm i     = go (i + 1) (f acc (prefix + i))
        | otherwise        = go (i + 1) acc
