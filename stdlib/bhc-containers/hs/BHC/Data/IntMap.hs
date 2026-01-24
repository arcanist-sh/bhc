-- |
-- Module      : BHC.Data.IntMap
-- Description : Efficient maps from Int keys to values
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- An efficient implementation of maps from integer keys to values
-- using Patricia tries (also known as radix trees).
--
-- This module is designed to be imported qualified:
--
-- @
-- import qualified BHC.Data.IntMap as IM
-- @

{-# LANGUAGE BangPatterns #-}

module BHC.Data.IntMap (
    -- * Map type
    IntMap,

    -- * Construction
    empty,
    singleton,
    fromList,
    fromListWith,
    fromListWithKey,
    fromAscList,
    fromDistinctAscList,

    -- * Insertion
    insert,
    insertWith,
    insertWithKey,
    insertLookupWithKey,

    -- * Deletion/Update
    delete,
    adjust,
    adjustWithKey,
    update,
    updateWithKey,
    updateLookupWithKey,
    alter,
    alterF,

    -- * Query
    lookup,
    (!?),
    (!),
    findWithDefault,
    member,
    notMember,
    null,
    size,

    -- * Combine
    union,
    unionWith,
    unionWithKey,
    unions,
    unionsWith,
    difference,
    differenceWith,
    differenceWithKey,
    intersection,
    intersectionWith,
    intersectionWithKey,

    -- * Traversal
    map,
    mapWithKey,
    traverseWithKey,
    mapAccum,
    mapAccumWithKey,
    mapAccumRWithKey,

    -- * Folds
    foldr,
    foldl,
    foldrWithKey,
    foldlWithKey,
    foldMapWithKey,

    -- * Strict folds
    foldr',
    foldl',
    foldrWithKey',
    foldlWithKey',

    -- * Conversion
    elems,
    keys,
    assocs,
    keysSet,
    toList,
    toAscList,
    toDescList,

    -- * Filter
    filter,
    filterWithKey,
    restrictKeys,
    withoutKeys,
    partition,
    partitionWithKey,
    mapMaybe,
    mapMaybeWithKey,
    mapEither,
    mapEitherWithKey,

    -- * Submap
    isSubmapOf,
    isSubmapOfBy,
    isProperSubmapOf,
    isProperSubmapOfBy,

    -- * Min/Max
    lookupMin,
    lookupMax,
    findMin,
    findMax,
    deleteMin,
    deleteMax,
    deleteFindMin,
    deleteFindMax,
    updateMin,
    updateMax,
    updateMinWithKey,
    updateMaxWithKey,
    minView,
    maxView,
    minViewWithKey,
    maxViewWithKey,

    -- * Split
    split,
    splitLookup,
    splitRoot,
) where

import Prelude hiding (lookup, map, filter, foldr, foldl, null)
import qualified Prelude as P
import Data.Bits
import Data.Maybe (fromMaybe)
import qualified BHC.Data.IntSet as IntSet

-- ============================================================
-- Type
-- ============================================================

-- | A map from 'Int' keys to values.
--
-- Uses Patricia tries for O(min(n, W)) operations where W is the
-- number of bits in an 'Int' (typically 64).
data IntMap a
    = Nil
    | Tip {-# UNPACK #-} !Key !a
    | Bin {-# UNPACK #-} !Prefix
          {-# UNPACK #-} !Mask
          !(IntMap a)
          !(IntMap a)
    deriving (Eq, Ord)

type Key = Int
type Prefix = Int
type Mask = Int

instance Show a => Show (IntMap a) where
    showsPrec d m = showParen (d > 10) $
        showString "fromList " . shows (toList m)

instance Functor IntMap where
    fmap = map

instance Foldable IntMap where
    foldr = foldr
    foldl = foldl
    null = null
    length = size

instance Traversable IntMap where
    traverse f = traverseWithKey (\_ x -> f x)

instance Semigroup (IntMap a) where
    (<>) = union

instance Monoid (IntMap a) where
    mempty = empty
    mappend = (<>)

-- ============================================================
-- Construction
-- ============================================================

-- | The empty map.
--
-- > empty == fromList []
-- > size empty == 0
empty :: IntMap a
empty = Nil

-- | A map with a single element.
--
-- > singleton 1 'a' == fromList [(1, 'a')]
-- > size (singleton 1 'a') == 1
singleton :: Key -> a -> IntMap a
singleton k x = Tip k x

-- | Build a map from a list of key/value pairs.
--
-- If the list contains duplicate keys, the last value for the key wins.
--
-- > fromList [] == empty
-- > fromList [(5,"a"), (3,"b"), (5,"c")] == fromList [(5,"c"), (3,"b")]
fromList :: [(Key, a)] -> IntMap a
fromList = P.foldl (\m (k, v) -> insert k v m) empty

-- | Build a map from a list with a combining function.
fromListWith :: (a -> a -> a) -> [(Key, a)] -> IntMap a
fromListWith f = fromListWithKey (\_ x y -> f x y)

-- | Build a map from a list with a combining function.
fromListWithKey :: (Key -> a -> a -> a) -> [(Key, a)] -> IntMap a
fromListWithKey f = P.foldl (\m (k, v) -> insertWithKey f k v m) empty

-- | Build a map from an ascending list.
fromAscList :: [(Key, a)] -> IntMap a
fromAscList = fromList

-- | Build a map from an ascending list of distinct keys.
fromDistinctAscList :: [(Key, a)] -> IntMap a
fromDistinctAscList = fromList

-- ============================================================
-- Insertion
-- ============================================================

-- | Insert a new key/value pair.
--
-- If the key is already present, the old value is replaced.
insert :: Key -> a -> IntMap a -> IntMap a
insert !k x = insertWithKey (\_ v _ -> v) k x

-- | Insert with a combining function.
insertWith :: (a -> a -> a) -> Key -> a -> IntMap a -> IntMap a
insertWith f = insertWithKey (\_ x y -> f x y)

-- | Insert with a combining function that has access to the key.
insertWithKey :: (Key -> a -> a -> a) -> Key -> a -> IntMap a -> IntMap a
insertWithKey f !k x t = case t of
    Nil -> Tip k x
    Tip ky y
        | k == ky   -> Tip k (f k x y)
        | otherwise -> link k (Tip k x) ky t
    Bin p m l r
        | nomatch k p m -> link k (Tip k x) p t
        | zero k m      -> Bin p m (insertWithKey f k x l) r
        | otherwise     -> Bin p m l (insertWithKey f k x r)

-- | Insert with a combining function, also returning the old value if present.
insertLookupWithKey :: (Key -> a -> a -> a) -> Key -> a -> IntMap a -> (Maybe a, IntMap a)
insertLookupWithKey f !k x t = case t of
    Nil -> (Nothing, Tip k x)
    Tip ky y
        | k == ky   -> (Just y, Tip k (f k x y))
        | otherwise -> (Nothing, link k (Tip k x) ky t)
    Bin p m l r
        | nomatch k p m -> (Nothing, link k (Tip k x) p t)
        | zero k m      -> let (found, l') = insertLookupWithKey f k x l
                           in (found, Bin p m l' r)
        | otherwise     -> let (found, r') = insertLookupWithKey f k x r
                           in (found, Bin p m l r')

-- ============================================================
-- Deletion/Update
-- ============================================================

-- | Delete a key from the map.
delete :: Key -> IntMap a -> IntMap a
delete !k t = case t of
    Nil -> Nil
    Tip ky _
        | k == ky   -> Nil
        | otherwise -> t
    Bin p m l r
        | nomatch k p m -> t
        | zero k m      -> bin p m (delete k l) r
        | otherwise     -> bin p m l (delete k r)

-- | Adjust a value at a specific key.
adjust :: (a -> a) -> Key -> IntMap a -> IntMap a
adjust f = adjustWithKey (\_ x -> f x)

-- | Adjust a value with access to the key.
adjustWithKey :: (Key -> a -> a) -> Key -> IntMap a -> IntMap a
adjustWithKey f = updateWithKey (\k x -> Just (f k x))

-- | Update a value at a specific key.
update :: (a -> Maybe a) -> Key -> IntMap a -> IntMap a
update f = updateWithKey (\_ x -> f x)

-- | Update with access to the key.
updateWithKey :: (Key -> a -> Maybe a) -> Key -> IntMap a -> IntMap a
updateWithKey f !k t = case t of
    Nil -> Nil
    Tip ky y
        | k == ky   -> case f k y of
            Nothing -> Nil
            Just y' -> Tip k y'
        | otherwise -> t
    Bin p m l r
        | nomatch k p m -> t
        | zero k m      -> bin p m (updateWithKey f k l) r
        | otherwise     -> bin p m l (updateWithKey f k r)

-- | Update with access to the key, also returning the old value.
updateLookupWithKey :: (Key -> a -> Maybe a) -> Key -> IntMap a -> (Maybe a, IntMap a)
updateLookupWithKey f !k t = case t of
    Nil -> (Nothing, Nil)
    Tip ky y
        | k == ky   -> case f k y of
            Nothing -> (Just y, Nil)
            Just y' -> (Just y, Tip k y')
        | otherwise -> (Nothing, t)
    Bin p m l r
        | nomatch k p m -> (Nothing, t)
        | zero k m      -> let (found, l') = updateLookupWithKey f k l
                           in (found, bin p m l' r)
        | otherwise     -> let (found, r') = updateLookupWithKey f k r
                           in (found, bin p m l r')

-- | Alter a value at a key.
alter :: (Maybe a -> Maybe a) -> Key -> IntMap a -> IntMap a
alter f !k t = case t of
    Nil -> case f Nothing of
        Nothing -> Nil
        Just x  -> Tip k x
    Tip ky y
        | k == ky   -> case f (Just y) of
            Nothing -> Nil
            Just y' -> Tip k y'
        | otherwise -> case f Nothing of
            Nothing -> t
            Just x  -> link k (Tip k x) ky t
    Bin p m l r
        | nomatch k p m -> case f Nothing of
            Nothing -> t
            Just x  -> link k (Tip k x) p t
        | zero k m      -> bin p m (alter f k l) r
        | otherwise     -> bin p m l (alter f k r)

-- | Alter with a functor.
alterF :: Functor f => (Maybe a -> f (Maybe a)) -> Key -> IntMap a -> f (IntMap a)
alterF f k m = fmap (\v -> case v of
    Nothing -> delete k m
    Just x  -> insert k x m) (f (lookup k m))

-- ============================================================
-- Query
-- ============================================================

-- | Lookup a value at a key.
lookup :: Key -> IntMap a -> Maybe a
lookup !k t = case t of
    Nil -> Nothing
    Tip ky y
        | k == ky   -> Just y
        | otherwise -> Nothing
    Bin _ m l r
        | zero k m  -> lookup k l
        | otherwise -> lookup k r

-- | Lookup operator.
(!?) :: IntMap a -> Key -> Maybe a
(!?) = flip lookup

-- | Lookup operator that throws on missing key.
(!) :: IntMap a -> Key -> a
m ! k = fromMaybe (P.error "IntMap.!: key not found") (lookup k m)

-- | Lookup with a default value.
findWithDefault :: a -> Key -> IntMap a -> a
findWithDefault def k m = fromMaybe def (lookup k m)

-- | Is the key a member of the map?
member :: Key -> IntMap a -> Bool
member k m = case lookup k m of
    Nothing -> False
    Just _  -> True

-- | Is the key not a member of the map?
notMember :: Key -> IntMap a -> Bool
notMember k = not . member k

-- | Is the map empty?
null :: IntMap a -> Bool
null Nil = True
null _   = False

-- | Number of elements in the map.
size :: IntMap a -> Int
size = go 0
  where
    go !acc Nil           = acc
    go !acc (Tip _ _)     = acc + 1
    go !acc (Bin _ _ l r) = go (go acc l) r

-- ============================================================
-- Combine
-- ============================================================

-- | Left-biased union.
union :: IntMap a -> IntMap a -> IntMap a
union = unionWith const

-- | Union with a combining function.
unionWith :: (a -> a -> a) -> IntMap a -> IntMap a -> IntMap a
unionWith f = unionWithKey (\_ x y -> f x y)

-- | Union with a combining function.
unionWithKey :: (Key -> a -> a -> a) -> IntMap a -> IntMap a -> IntMap a
unionWithKey f t1 t2 = case (t1, t2) of
    (Nil, t) -> t
    (t, Nil) -> t
    (Tip k x, t) -> insertWithKey f k x t
    (t, Tip k x) -> insertWithKey (\k' x' y -> f k' y x') k x t
    (Bin p1 m1 l1 r1, Bin p2 m2 l2 r2)
        | shorter m1 m2 -> union1
        | shorter m2 m1 -> union2
        | p1 == p2      -> Bin p1 m1 (unionWithKey f l1 l2) (unionWithKey f r1 r2)
        | otherwise     -> link p1 t1 p2 t2
      where
        union1
            | nomatch p2 p1 m1 = link p1 t1 p2 t2
            | zero p2 m1       = Bin p1 m1 (unionWithKey f l1 t2) r1
            | otherwise        = Bin p1 m1 l1 (unionWithKey f r1 t2)
        union2
            | nomatch p1 p2 m2 = link p1 t1 p2 t2
            | zero p1 m2       = Bin p2 m2 (unionWithKey f t1 l2) r2
            | otherwise        = Bin p2 m2 l2 (unionWithKey f t1 r2)

-- | Union of a list of maps.
unions :: [IntMap a] -> IntMap a
unions = P.foldl union empty

-- | Union of a list with a combining function.
unionsWith :: (a -> a -> a) -> [IntMap a] -> IntMap a
unionsWith f = P.foldl (unionWith f) empty

-- | Difference of two maps.
difference :: IntMap a -> IntMap b -> IntMap a
difference = differenceWith (\_ _ -> Nothing)

-- | Difference with a combining function.
differenceWith :: (a -> b -> Maybe a) -> IntMap a -> IntMap b -> IntMap a
differenceWith f = differenceWithKey (\_ x y -> f x y)

-- | Difference with a combining function.
differenceWithKey :: (Key -> a -> b -> Maybe a) -> IntMap a -> IntMap b -> IntMap a
differenceWithKey f t1 t2 = case (t1, t2) of
    (Nil, _)   -> Nil
    (t, Nil)   -> t
    (Tip k x, t) -> case lookup k t of
        Nothing -> Tip k x
        Just y  -> case f k x y of
            Nothing -> Nil
            Just z  -> Tip k z
    (t, Tip k y) -> updateWithKey (\k' x -> f k' x y) k t
    (Bin p1 m1 l1 r1, Bin p2 m2 l2 r2)
        | shorter m1 m2 -> diff1
        | shorter m2 m1 -> diff2
        | p1 == p2      -> bin p1 m1 (differenceWithKey f l1 l2) (differenceWithKey f r1 r2)
        | otherwise     -> t1
      where
        diff1
            | nomatch p2 p1 m1 = t1
            | zero p2 m1       = bin p1 m1 (differenceWithKey f l1 t2) r1
            | otherwise        = bin p1 m1 l1 (differenceWithKey f r1 t2)
        diff2
            | nomatch p1 p2 m2 = t1
            | zero p1 m2       = differenceWithKey f t1 l2
            | otherwise        = differenceWithKey f t1 r2

-- | Intersection of two maps.
intersection :: IntMap a -> IntMap b -> IntMap a
intersection = intersectionWith const

-- | Intersection with a combining function.
intersectionWith :: (a -> b -> c) -> IntMap a -> IntMap b -> IntMap c
intersectionWith f = intersectionWithKey (\_ x y -> f x y)

-- | Intersection with a combining function.
intersectionWithKey :: (Key -> a -> b -> c) -> IntMap a -> IntMap b -> IntMap c
intersectionWithKey f t1 t2 = case (t1, t2) of
    (Nil, _)   -> Nil
    (_, Nil)   -> Nil
    (Tip k x, t) -> case lookup k t of
        Nothing -> Nil
        Just y  -> Tip k (f k x y)
    (t, Tip k y) -> case lookup k t of
        Nothing -> Nil
        Just x  -> Tip k (f k x y)
    (Bin p1 m1 l1 r1, Bin p2 m2 l2 r2)
        | shorter m1 m2 -> inter1
        | shorter m2 m1 -> inter2
        | p1 == p2      -> bin p1 m1 (intersectionWithKey f l1 l2) (intersectionWithKey f r1 r2)
        | otherwise     -> Nil
      where
        inter1
            | nomatch p2 p1 m1 = Nil
            | zero p2 m1       = intersectionWithKey f l1 t2
            | otherwise        = intersectionWithKey f r1 t2
        inter2
            | nomatch p1 p2 m2 = Nil
            | zero p1 m2       = intersectionWithKey f t1 l2
            | otherwise        = intersectionWithKey f t1 r2

-- ============================================================
-- Traversal
-- ============================================================

-- | Map a function over values.
map :: (a -> b) -> IntMap a -> IntMap b
map f = mapWithKey (\_ x -> f x)

-- | Map with access to the key.
mapWithKey :: (Key -> a -> b) -> IntMap a -> IntMap b
mapWithKey f t = case t of
    Nil           -> Nil
    Tip k x       -> Tip k (f k x)
    Bin p m l r   -> Bin p m (mapWithKey f l) (mapWithKey f r)

-- | Traverse with access to keys.
traverseWithKey :: Applicative f => (Key -> a -> f b) -> IntMap a -> f (IntMap b)
traverseWithKey f t = case t of
    Nil           -> pure Nil
    Tip k x       -> Tip k <$> f k x
    Bin p m l r   -> Bin p m <$> traverseWithKey f l <*> traverseWithKey f r

-- | Map with an accumulator.
mapAccum :: (a -> b -> (a, c)) -> a -> IntMap b -> (a, IntMap c)
mapAccum f = mapAccumWithKey (\a _ x -> f a x)

-- | Map with an accumulator and key access.
mapAccumWithKey :: (a -> Key -> b -> (a, c)) -> a -> IntMap b -> (a, IntMap c)
mapAccumWithKey f acc t = case t of
    Nil           -> (acc, Nil)
    Tip k x       -> let (acc', y) = f acc k x in (acc', Tip k y)
    Bin p m l r   -> let (acc1, l') = mapAccumWithKey f acc l
                         (acc2, r') = mapAccumWithKey f acc1 r
                     in (acc2, Bin p m l' r')

-- | Map accumulating right-to-left.
mapAccumRWithKey :: (a -> Key -> b -> (a, c)) -> a -> IntMap b -> (a, IntMap c)
mapAccumRWithKey f acc t = case t of
    Nil           -> (acc, Nil)
    Tip k x       -> let (acc', y) = f acc k x in (acc', Tip k y)
    Bin p m l r   -> let (acc1, r') = mapAccumRWithKey f acc r
                         (acc2, l') = mapAccumRWithKey f acc1 l
                     in (acc2, Bin p m l' r')

-- ============================================================
-- Folds
-- ============================================================

-- | Lazy right fold.
foldr :: (a -> b -> b) -> b -> IntMap a -> b
foldr f z = foldrWithKey (\_ x acc -> f x acc) z

-- | Lazy left fold.
foldl :: (b -> a -> b) -> b -> IntMap a -> b
foldl f z = foldlWithKey (\acc _ x -> f acc x) z

-- | Lazy right fold with key.
foldrWithKey :: (Key -> a -> b -> b) -> b -> IntMap a -> b
foldrWithKey f z t = case t of
    Nil           -> z
    Tip k x       -> f k x z
    Bin _ _ l r   -> foldrWithKey f (foldrWithKey f z r) l

-- | Lazy left fold with key.
foldlWithKey :: (b -> Key -> a -> b) -> b -> IntMap a -> b
foldlWithKey f z t = case t of
    Nil           -> z
    Tip k x       -> f z k x
    Bin _ _ l r   -> foldlWithKey f (foldlWithKey f z l) r

-- | Fold to a monoid with key.
foldMapWithKey :: Monoid m => (Key -> a -> m) -> IntMap a -> m
foldMapWithKey f = foldrWithKey (\k x acc -> f k x <> acc) mempty

-- | Strict right fold.
foldr' :: (a -> b -> b) -> b -> IntMap a -> b
foldr' f z = foldrWithKey' (\_ x acc -> f x acc) z

-- | Strict left fold.
foldl' :: (b -> a -> b) -> b -> IntMap a -> b
foldl' f z = foldlWithKey' (\acc _ x -> f acc x) z

-- | Strict right fold with key.
foldrWithKey' :: (Key -> a -> b -> b) -> b -> IntMap a -> b
foldrWithKey' f !z t = case t of
    Nil           -> z
    Tip k x       -> f k x z
    Bin _ _ l r   -> let !z' = foldrWithKey' f z r
                     in foldrWithKey' f z' l

-- | Strict left fold with key.
foldlWithKey' :: (b -> Key -> a -> b) -> b -> IntMap a -> b
foldlWithKey' f !z t = case t of
    Nil           -> z
    Tip k x       -> f z k x
    Bin _ _ l r   -> let !z' = foldlWithKey' f z l
                     in foldlWithKey' f z' r

-- ============================================================
-- Conversion
-- ============================================================

-- | All values in ascending key order.
elems :: IntMap a -> [a]
elems = foldr (:) []

-- | All keys in ascending order.
keys :: IntMap a -> [Key]
keys = foldrWithKey (\k _ ks -> k : ks) []

-- | All key/value pairs in ascending key order.
assocs :: IntMap a -> [(Key, a)]
assocs = toAscList

-- | Convert keys to an IntSet.
keysSet :: IntMap a -> IntSet.IntSet
keysSet = foldrWithKey (\k _ s -> IntSet.insert k s) IntSet.empty

-- | Convert to a list.
toList :: IntMap a -> [(Key, a)]
toList = toAscList

-- | Convert to an ascending list.
toAscList :: IntMap a -> [(Key, a)]
toAscList = foldrWithKey (\k x xs -> (k, x) : xs) []

-- | Convert to a descending list.
toDescList :: IntMap a -> [(Key, a)]
toDescList = foldlWithKey (\xs k x -> (k, x) : xs) []

-- ============================================================
-- Filter
-- ============================================================

-- | Filter values satisfying a predicate.
filter :: (a -> Bool) -> IntMap a -> IntMap a
filter p = filterWithKey (\_ x -> p x)

-- | Filter with key access.
filterWithKey :: (Key -> a -> Bool) -> IntMap a -> IntMap a
filterWithKey p t = case t of
    Nil           -> Nil
    Tip k x
        | p k x     -> t
        | otherwise -> Nil
    Bin pr m l r  -> bin pr m (filterWithKey p l) (filterWithKey p r)

-- | Restrict to keys in a set.
restrictKeys :: IntMap a -> IntSet.IntSet -> IntMap a
restrictKeys m s = filterWithKey (\k _ -> IntSet.member k s) m

-- | Remove keys in a set.
withoutKeys :: IntMap a -> IntSet.IntSet -> IntMap a
withoutKeys m s = filterWithKey (\k _ -> not (IntSet.member k s)) m

-- | Partition by a predicate.
partition :: (a -> Bool) -> IntMap a -> (IntMap a, IntMap a)
partition p = partitionWithKey (\_ x -> p x)

-- | Partition with key access.
partitionWithKey :: (Key -> a -> Bool) -> IntMap a -> (IntMap a, IntMap a)
partitionWithKey p t = (filterWithKey p t, filterWithKey (\k x -> not (p k x)) t)

-- | Map and collect Just results.
mapMaybe :: (a -> Maybe b) -> IntMap a -> IntMap b
mapMaybe f = mapMaybeWithKey (\_ x -> f x)

-- | Map and collect Just results with key access.
mapMaybeWithKey :: (Key -> a -> Maybe b) -> IntMap a -> IntMap b
mapMaybeWithKey f t = case t of
    Nil           -> Nil
    Tip k x       -> case f k x of
        Nothing -> Nil
        Just y  -> Tip k y
    Bin p m l r   -> bin p m (mapMaybeWithKey f l) (mapMaybeWithKey f r)

-- | Map and partition by Left/Right.
mapEither :: (a -> Either b c) -> IntMap a -> (IntMap b, IntMap c)
mapEither f = mapEitherWithKey (\_ x -> f x)

-- | Map and partition with key access.
mapEitherWithKey :: (Key -> a -> Either b c) -> IntMap a -> (IntMap b, IntMap c)
mapEitherWithKey f t = (mapMaybeWithKey (\k x -> either Just (const Nothing) (f k x)) t,
                        mapMaybeWithKey (\k x -> either (const Nothing) Just (f k x)) t)

-- ============================================================
-- Submap
-- ============================================================

-- | Is the first map a submap of the second?
isSubmapOf :: Eq a => IntMap a -> IntMap a -> Bool
isSubmapOf = isSubmapOfBy (==)

-- | Submap with custom equality.
isSubmapOfBy :: (a -> b -> Bool) -> IntMap a -> IntMap b -> Bool
isSubmapOfBy eq t1 t2 = size t1 <= size t2 && go t1 t2
  where
    go Nil _ = True
    go _ Nil = False
    go (Tip k x) t = case lookup k t of
        Nothing -> False
        Just y  -> eq x y
    go (Bin p1 m1 l1 r1) (Bin p2 m2 l2 r2)
        | shorter m1 m2 = False
        | shorter m2 m1 = match p1 p2 m2 &&
                          (if zero p1 m2 then go t1 l2 else go t1 r2)
        | otherwise     = p1 == p2 && go l1 l2 && go r1 r2
      where t1 = Bin p1 m1 l1 r1

-- | Is the first map a proper submap of the second?
isProperSubmapOf :: Eq a => IntMap a -> IntMap a -> Bool
isProperSubmapOf = isProperSubmapOfBy (==)

-- | Proper submap with custom equality.
isProperSubmapOfBy :: (a -> b -> Bool) -> IntMap a -> IntMap b -> Bool
isProperSubmapOfBy eq t1 t2 = size t1 < size t2 && isSubmapOfBy eq t1 t2

-- ============================================================
-- Min/Max
-- ============================================================

-- | Lookup minimum key and value.
lookupMin :: IntMap a -> Maybe (Key, a)
lookupMin Nil             = Nothing
lookupMin (Tip k x)       = Just (k, x)
lookupMin (Bin _ _ l _)   = lookupMin l

-- | Lookup maximum key and value.
lookupMax :: IntMap a -> Maybe (Key, a)
lookupMax Nil             = Nothing
lookupMax (Tip k x)       = Just (k, x)
lookupMax (Bin _ _ _ r)   = lookupMax r

-- | Find minimum (partial).
findMin :: IntMap a -> (Key, a)
findMin = fromMaybe (P.error "IntMap.findMin: empty map") . lookupMin

-- | Find maximum (partial).
findMax :: IntMap a -> (Key, a)
findMax = fromMaybe (P.error "IntMap.findMax: empty map") . lookupMax

-- | Delete minimum.
deleteMin :: IntMap a -> IntMap a
deleteMin = maybe empty snd . minView

-- | Delete maximum.
deleteMax :: IntMap a -> IntMap a
deleteMax = maybe empty snd . maxView

-- | Delete and return minimum.
deleteFindMin :: IntMap a -> ((Key, a), IntMap a)
deleteFindMin m = case minViewWithKey m of
    Nothing      -> P.error "IntMap.deleteFindMin: empty map"
    Just (kv, m') -> (kv, m')

-- | Delete and return maximum.
deleteFindMax :: IntMap a -> ((Key, a), IntMap a)
deleteFindMax m = case maxViewWithKey m of
    Nothing      -> P.error "IntMap.deleteFindMax: empty map"
    Just (kv, m') -> (kv, m')

-- | Update minimum value.
updateMin :: (a -> Maybe a) -> IntMap a -> IntMap a
updateMin f = updateMinWithKey (\_ x -> f x)

-- | Update maximum value.
updateMax :: (a -> Maybe a) -> IntMap a -> IntMap a
updateMax f = updateMaxWithKey (\_ x -> f x)

-- | Update minimum with key.
updateMinWithKey :: (Key -> a -> Maybe a) -> IntMap a -> IntMap a
updateMinWithKey f t = case t of
    Nil           -> Nil
    Tip k x       -> case f k x of
        Nothing -> Nil
        Just y  -> Tip k y
    Bin p m l r   -> bin p m (updateMinWithKey f l) r

-- | Update maximum with key.
updateMaxWithKey :: (Key -> a -> Maybe a) -> IntMap a -> IntMap a
updateMaxWithKey f t = case t of
    Nil           -> Nil
    Tip k x       -> case f k x of
        Nothing -> Nil
        Just y  -> Tip k y
    Bin p m l r   -> bin p m l (updateMaxWithKey f r)

-- | View with minimum removed.
minView :: IntMap a -> Maybe (a, IntMap a)
minView = fmap (\((_, x), m) -> (x, m)) . minViewWithKey

-- | View with maximum removed.
maxView :: IntMap a -> Maybe (a, IntMap a)
maxView = fmap (\((_, x), m) -> (x, m)) . maxViewWithKey

-- | View with minimum key/value removed.
minViewWithKey :: IntMap a -> Maybe ((Key, a), IntMap a)
minViewWithKey t = case t of
    Nil           -> Nothing
    Tip k x       -> Just ((k, x), Nil)
    Bin p m l r   -> case minViewWithKey l of
        Nothing          -> minViewWithKey r
        Just (kv, l')    -> Just (kv, bin p m l' r)

-- | View with maximum key/value removed.
maxViewWithKey :: IntMap a -> Maybe ((Key, a), IntMap a)
maxViewWithKey t = case t of
    Nil           -> Nothing
    Tip k x       -> Just ((k, x), Nil)
    Bin p m l r   -> case maxViewWithKey r of
        Nothing          -> maxViewWithKey l
        Just (kv, r')    -> Just (kv, bin p m l r')

-- ============================================================
-- Split
-- ============================================================

-- | Split at a key.
split :: Key -> IntMap a -> (IntMap a, IntMap a)
split k m = case splitLookup k m of
    (lt, _, gt) -> (lt, gt)

-- | Split at a key, also returning the value if present.
splitLookup :: Key -> IntMap a -> (IntMap a, Maybe a, IntMap a)
splitLookup k t = case t of
    Nil -> (Nil, Nothing, Nil)
    Tip ky y
        | k > ky    -> (t, Nothing, Nil)
        | k < ky    -> (Nil, Nothing, t)
        | otherwise -> (Nil, Just y, Nil)
    Bin p m l r
        | nomatch k p m -> if k > p then (t, Nothing, Nil) else (Nil, Nothing, t)
        | zero k m      -> let (lt, found, gt) = splitLookup k l
                           in (lt, found, bin p m gt r)
        | otherwise     -> let (lt, found, gt) = splitLookup k r
                           in (bin p m l lt, found, gt)

-- | Decompose into pieces based on structure.
splitRoot :: IntMap a -> [IntMap a]
splitRoot Nil           = []
splitRoot (Tip k x)     = [Tip k x]
splitRoot (Bin _ _ l r) = [l, r]

-- ============================================================
-- Internal Helpers
-- ============================================================

-- | Link two disjoint trees.
link :: Prefix -> IntMap a -> Prefix -> IntMap a -> IntMap a
link p1 t1 p2 t2
    | zero p1 m = Bin p m t1 t2
    | otherwise = Bin p m t2 t1
  where
    m = branchMask p1 p2
    p = mask p1 m

-- | Smart constructor.
bin :: Prefix -> Mask -> IntMap a -> IntMap a -> IntMap a
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

-- | Is the first mask shorter (higher bit)?
shorter :: Mask -> Mask -> Bool
shorter m1 m2 = (m1 > m2)

-- | Find branching bit.
branchMask :: Prefix -> Prefix -> Mask
branchMask p1 p2 = highestBitMask (p1 `xor` p2)

-- | Get highest set bit as mask.
highestBitMask :: Int -> Int
highestBitMask x = x' `xor` (x' `shiftR` 1)
  where
    x' = foldr (.|.) x [shiftR x i | i <- [1, 2, 4, 8, 16, 32]]
