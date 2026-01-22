-- |
-- Module      : BHC.Data.Map
-- Description : Ordered maps from keys to values
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- An efficient implementation of ordered maps from keys to values.

module BHC.Data.Map (
    -- * Map type
    Map,
    
    -- * Construction
    empty, singleton, fromList, fromListWith, fromListWithKey,
    
    -- * Insertion
    insert, insertWith, insertWithKey, insertLookupWithKey,
    
    -- * Deletion/Update
    delete, adjust, adjustWithKey, update, updateWithKey,
    alter, alterF,
    
    -- * Query
    lookup, (!?), (!), findWithDefault,
    member, notMember,
    null, size,
    
    -- * Combine
    union, unionWith, unionWithKey,
    unions, unionsWith,
    difference, differenceWith, differenceWithKey,
    intersection, intersectionWith, intersectionWithKey,
    
    -- * Traversal
    map, mapWithKey, traverseWithKey,
    mapAccum, mapAccumWithKey,
    mapKeys, mapKeysWith, mapKeysMonotonic,
    
    -- * Folds
    foldr, foldl, foldrWithKey, foldlWithKey,
    foldMapWithKey,
    
    -- * Conversion
    elems, keys, assocs, keysSet,
    toList, toAscList, toDescList,
    
    -- * Filter
    filter, filterWithKey,
    partition, partitionWithKey,
    mapMaybe, mapMaybeWithKey,
    mapEither, mapEitherWithKey,
    
    -- * Submap
    isSubmapOf, isSubmapOfBy,
    isProperSubmapOf, isProperSubmapOfBy,
    
    -- * Min/Max
    lookupMin, lookupMax,
    findMin, findMax,
    deleteMin, deleteMax,
    
    -- * Split
    split, splitLookup,
) where

import BHC.Prelude hiding (map, lookup, null, filter, foldr, foldl)
import qualified BHC.Prelude as P
import qualified BHC.Data.Set as Set

-- | A map from keys @k@ to values @a@.
data Map k a
    = Tip
    | Bin {-# UNPACK #-} !Int !k a !(Map k a) !(Map k a)
    deriving (Eq, Ord, Show, Read)

instance Functor (Map k) where
    fmap = map

instance Foldable (Map k) where
    foldr = foldr
    foldl = foldl
    null = null
    length = size

instance Traversable (Map k) where
    traverse f = traverseWithKey (const f)

instance (Ord k, Semigroup a) => Semigroup (Map k a) where
    (<>) = unionWith (<>)

instance (Ord k, Semigroup a) => Monoid (Map k a) where
    mempty = empty

-- Construction
empty :: Map k a
empty = Tip

singleton :: k -> a -> Map k a
singleton k x = Bin 1 k x Tip Tip

fromList :: Ord k => [(k, a)] -> Map k a
fromList = P.foldl' (\m (k, x) -> insert k x m) empty

fromListWith :: Ord k => (a -> a -> a) -> [(k, a)] -> Map k a
fromListWith f = fromListWithKey (\_ x y -> f x y)

fromListWithKey :: Ord k => (k -> a -> a -> a) -> [(k, a)] -> Map k a
fromListWithKey f = P.foldl' ins empty
  where ins m (k, x) = insertWithKey f k x m

-- Query
lookup :: Ord k => k -> Map k a -> Maybe a
lookup = go
  where
    go _ Tip = Nothing
    go k (Bin _ kx x l r) = case compare k kx of
        LT -> go k l
        GT -> go k r
        EQ -> Just x

(!?) :: Ord k => Map k a -> k -> Maybe a
(!?) = flip lookup
infixl 9 !?

(!) :: Ord k => Map k a -> k -> a
m ! k = case lookup k m of
    Just x  -> x
    Nothing -> error "Map.!: key not found"
infixl 9 !

findWithDefault :: Ord k => a -> k -> Map k a -> a
findWithDefault def k m = case lookup k m of
    Just x  -> x
    Nothing -> def

member :: Ord k => k -> Map k a -> Bool
member k m = case lookup k m of
    Just _  -> True
    Nothing -> False

notMember :: Ord k => k -> Map k a -> Bool
notMember k = not . member k

null :: Map k a -> Bool
null Tip = True
null _   = False

size :: Map k a -> Int
size Tip            = 0
size (Bin n _ _ _ _) = n

-- Insertion
insert :: Ord k => k -> a -> Map k a -> Map k a
insert = insertWith const

insertWith :: Ord k => (a -> a -> a) -> k -> a -> Map k a -> Map k a
insertWith f = insertWithKey (\_ x y -> f x y)

insertWithKey :: Ord k => (k -> a -> a -> a) -> k -> a -> Map k a -> Map k a
insertWithKey f k x = go
  where
    go Tip = singleton k x
    go (Bin sz ky y l r) = case compare k ky of
        LT -> balance ky y (go l) r
        GT -> balance ky y l (go r)
        EQ -> Bin sz k (f k x y) l r

insertLookupWithKey :: Ord k => (k -> a -> a -> a) -> k -> a -> Map k a -> (Maybe a, Map k a)
insertLookupWithKey f k x = go
  where
    go Tip = (Nothing, singleton k x)
    go (Bin sz ky y l r) = case compare k ky of
        LT -> let (found, l') = go l in (found, balance ky y l' r)
        GT -> let (found, r') = go r in (found, balance ky y l r')
        EQ -> (Just y, Bin sz k (f k x y) l r)

-- Deletion
delete :: Ord k => k -> Map k a -> Map k a
delete = go
  where
    go _ Tip = Tip
    go k (Bin _ kx x l r) = case compare k kx of
        LT -> balance kx x (go k l) r
        GT -> balance kx x l (go k r)
        EQ -> glue l r

adjust :: Ord k => (a -> a) -> k -> Map k a -> Map k a
adjust f = adjustWithKey (\_ x -> f x)

adjustWithKey :: Ord k => (k -> a -> a) -> k -> Map k a -> Map k a
adjustWithKey f = go
  where
    go _ Tip = Tip
    go k (Bin sx kx x l r) = case compare k kx of
        LT -> Bin sx kx x (go k l) r
        GT -> Bin sx kx x l (go k r)
        EQ -> Bin sx kx (f kx x) l r

update :: Ord k => (a -> Maybe a) -> k -> Map k a -> Map k a
update f = updateWithKey (\_ x -> f x)

updateWithKey :: Ord k => (k -> a -> Maybe a) -> k -> Map k a -> Map k a
updateWithKey f = go
  where
    go _ Tip = Tip
    go k (Bin _ kx x l r) = case compare k kx of
        LT -> balance kx x (go k l) r
        GT -> balance kx x l (go k r)
        EQ -> case f kx x of
            Just x' -> Bin (size l + size r + 1) kx x' l r
            Nothing -> glue l r

alter :: Ord k => (Maybe a -> Maybe a) -> k -> Map k a -> Map k a
alter f k = go
  where
    go Tip = case f Nothing of
        Nothing -> Tip
        Just x  -> singleton k x
    go (Bin sx kx x l r) = case compare k kx of
        LT -> balance kx x (go l) r
        GT -> balance kx x l (go r)
        EQ -> case f (Just x) of
            Just x' -> Bin sx kx x' l r
            Nothing -> glue l r

alterF :: (Functor f, Ord k) => (Maybe a -> f (Maybe a)) -> k -> Map k a -> f (Map k a)
alterF f k m = fmap ins (f (lookup k m))
  where ins Nothing  = delete k m
        ins (Just x) = insert k x m

-- Combine
union :: Ord k => Map k a -> Map k a -> Map k a
union = unionWith const

unionWith :: Ord k => (a -> a -> a) -> Map k a -> Map k a -> Map k a
unionWith f = unionWithKey (\_ x y -> f x y)

unionWithKey :: Ord k => (k -> a -> a -> a) -> Map k a -> Map k a -> Map k a
unionWithKey f t1 t2 = P.foldl' ins t1 (toList t2)
  where ins m (k, x) = insertWithKey f k x m

unions :: (Foldable f, Ord k) => f (Map k a) -> Map k a
unions = P.foldl' union empty

unionsWith :: (Foldable f, Ord k) => (a -> a -> a) -> f (Map k a) -> Map k a
unionsWith f = P.foldl' (unionWith f) empty

difference :: Ord k => Map k a -> Map k b -> Map k a
difference = differenceWith (\_ _ -> Nothing)

differenceWith :: Ord k => (a -> b -> Maybe a) -> Map k a -> Map k b -> Map k a
differenceWith f = differenceWithKey (\_ x y -> f x y)

differenceWithKey :: Ord k => (k -> a -> b -> Maybe a) -> Map k a -> Map k b -> Map k a
differenceWithKey f t1 t2 = filterWithKey check t1
  where check k x = case lookup k t2 of
            Nothing -> True
            Just y  -> case f k x y of
                Nothing -> False
                Just _  -> True

intersection :: Ord k => Map k a -> Map k b -> Map k a
intersection = intersectionWith const

intersectionWith :: Ord k => (a -> b -> c) -> Map k a -> Map k b -> Map k c
intersectionWith f = intersectionWithKey (\_ x y -> f x y)

intersectionWithKey :: Ord k => (k -> a -> b -> c) -> Map k a -> Map k b -> Map k c
intersectionWithKey f t1 t2 = mapMaybeWithKey go t1
  where go k x = case lookup k t2 of
            Nothing -> Nothing
            Just y  -> Just (f k x y)

-- Traversal
map :: (a -> b) -> Map k a -> Map k b
map f = mapWithKey (\_ x -> f x)

mapWithKey :: (k -> a -> b) -> Map k a -> Map k b
mapWithKey _ Tip = Tip
mapWithKey f (Bin sx kx x l r) = Bin sx kx (f kx x) (mapWithKey f l) (mapWithKey f r)

traverseWithKey :: Applicative t => (k -> a -> t b) -> Map k a -> t (Map k b)
traverseWithKey _ Tip = pure Tip
traverseWithKey f (Bin s k x l r) =
    (\l' x' r' -> Bin s k x' l' r') <$> traverseWithKey f l <*> f k x <*> traverseWithKey f r

mapAccum :: (a -> b -> (a, c)) -> a -> Map k b -> (a, Map k c)
mapAccum f = mapAccumWithKey (\a _ x -> f a x)

mapAccumWithKey :: (a -> k -> b -> (a, c)) -> a -> Map k b -> (a, Map k c)
mapAccumWithKey _ a Tip = (a, Tip)
mapAccumWithKey f a (Bin sx kx x l r) =
    let (a1, l') = mapAccumWithKey f a l
        (a2, x') = f a1 kx x
        (a3, r') = mapAccumWithKey f a2 r
    in (a3, Bin sx kx x' l' r')

mapKeys :: Ord k2 => (k1 -> k2) -> Map k1 a -> Map k2 a
mapKeys f = fromList . P.map (\(k, x) -> (f k, x)) . toList

mapKeysWith :: Ord k2 => (a -> a -> a) -> (k1 -> k2) -> Map k1 a -> Map k2 a
mapKeysWith c f = fromListWith c . P.map (\(k, x) -> (f k, x)) . toList

mapKeysMonotonic :: (k1 -> k2) -> Map k1 a -> Map k2 a
mapKeysMonotonic _ Tip = Tip
mapKeysMonotonic f (Bin sx kx x l r) =
    Bin sx (f kx) x (mapKeysMonotonic f l) (mapKeysMonotonic f r)

-- Folds
foldr :: (a -> b -> b) -> b -> Map k a -> b
foldr f = foldrWithKey (\_ x z -> f x z)

foldl :: (a -> b -> a) -> a -> Map k b -> a
foldl f = foldlWithKey (\z _ x -> f z x)

foldrWithKey :: (k -> a -> b -> b) -> b -> Map k a -> b
foldrWithKey _ z Tip = z
foldrWithKey f z (Bin _ kx x l r) = foldrWithKey f (f kx x (foldrWithKey f z r)) l

foldlWithKey :: (a -> k -> b -> a) -> a -> Map k b -> a
foldlWithKey _ z Tip = z
foldlWithKey f z (Bin _ kx x l r) = foldlWithKey f (f (foldlWithKey f z l) kx x) r

foldMapWithKey :: Monoid m => (k -> a -> m) -> Map k a -> m
foldMapWithKey f = foldrWithKey (\k x m -> f k x <> m) mempty

-- Conversion
elems :: Map k a -> [a]
elems = foldr (:) []

keys :: Map k a -> [k]
keys = foldrWithKey (\k _ ks -> k:ks) []

assocs :: Map k a -> [(k, a)]
assocs = toAscList

keysSet :: Map k a -> Set.Set k
keysSet = Set.fromList . keys

toList :: Map k a -> [(k, a)]
toList = toAscList

toAscList :: Map k a -> [(k, a)]
toAscList = foldrWithKey (\k x xs -> (k, x):xs) []

toDescList :: Map k a -> [(k, a)]
toDescList = foldlWithKey (\xs k x -> (k, x):xs) []

-- Filter
filter :: (a -> Bool) -> Map k a -> Map k a
filter p = filterWithKey (\_ x -> p x)

filterWithKey :: (k -> a -> Bool) -> Map k a -> Map k a
filterWithKey _ Tip = Tip
filterWithKey p (Bin _ kx x l r)
    | p kx x    = link kx x (filterWithKey p l) (filterWithKey p r)
    | otherwise = merge (filterWithKey p l) (filterWithKey p r)

partition :: (a -> Bool) -> Map k a -> (Map k a, Map k a)
partition p = partitionWithKey (\_ x -> p x)

partitionWithKey :: (k -> a -> Bool) -> Map k a -> (Map k a, Map k a)
partitionWithKey _ Tip = (Tip, Tip)
partitionWithKey p (Bin _ kx x l r)
    | p kx x    = (link kx x l1 r1, merge l2 r2)
    | otherwise = (merge l1 r1, link kx x l2 r2)
  where
    (l1, l2) = partitionWithKey p l
    (r1, r2) = partitionWithKey p r

mapMaybe :: (a -> Maybe b) -> Map k a -> Map k b
mapMaybe f = mapMaybeWithKey (\_ x -> f x)

mapMaybeWithKey :: (k -> a -> Maybe b) -> Map k a -> Map k b
mapMaybeWithKey _ Tip = Tip
mapMaybeWithKey f (Bin _ kx x l r) = case f kx x of
    Just y  -> link kx y (mapMaybeWithKey f l) (mapMaybeWithKey f r)
    Nothing -> merge (mapMaybeWithKey f l) (mapMaybeWithKey f r)

mapEither :: (a -> Either b c) -> Map k a -> (Map k b, Map k c)
mapEither f = mapEitherWithKey (\_ x -> f x)

mapEitherWithKey :: (k -> a -> Either b c) -> Map k a -> (Map k b, Map k c)
mapEitherWithKey _ Tip = (Tip, Tip)
mapEitherWithKey f (Bin _ kx x l r) = case f kx x of
    Left y  -> (link kx y l1 r1, merge l2 r2)
    Right z -> (merge l1 r1, link kx z l2 r2)
  where
    (l1, l2) = mapEitherWithKey f l
    (r1, r2) = mapEitherWithKey f r

-- Submap
isSubmapOf :: (Ord k, Eq a) => Map k a -> Map k a -> Bool
isSubmapOf = isSubmapOfBy (==)

isSubmapOfBy :: Ord k => (a -> b -> Bool) -> Map k a -> Map k b -> Bool
isSubmapOfBy f t1 t2 = P.all check (toList t1)
  where check (k, x) = case lookup k t2 of
            Nothing -> False
            Just y  -> f x y

isProperSubmapOf :: (Ord k, Eq a) => Map k a -> Map k a -> Bool
isProperSubmapOf = isProperSubmapOfBy (==)

isProperSubmapOfBy :: Ord k => (a -> b -> Bool) -> Map k a -> Map k b -> Bool
isProperSubmapOfBy f t1 t2 = size t1 < size t2 && isSubmapOfBy f t1 t2

-- Min/Max
lookupMin :: Map k a -> Maybe (k, a)
lookupMin Tip = Nothing
lookupMin (Bin _ k x Tip _) = Just (k, x)
lookupMin (Bin _ _ _ l _) = lookupMin l

lookupMax :: Map k a -> Maybe (k, a)
lookupMax Tip = Nothing
lookupMax (Bin _ k x _ Tip) = Just (k, x)
lookupMax (Bin _ _ _ _ r) = lookupMax r

findMin :: Map k a -> (k, a)
findMin m = case lookupMin m of
    Just kv -> kv
    Nothing -> error "Map.findMin: empty map"

findMax :: Map k a -> (k, a)
findMax m = case lookupMax m of
    Just kv -> kv
    Nothing -> error "Map.findMax: empty map"

deleteMin :: Map k a -> Map k a
deleteMin Tip = Tip
deleteMin (Bin _ _ _ Tip r) = r
deleteMin (Bin _ kx x l r) = balance kx x (deleteMin l) r

deleteMax :: Map k a -> Map k a
deleteMax Tip = Tip
deleteMax (Bin _ _ _ l Tip) = l
deleteMax (Bin _ kx x l r) = balance kx x l (deleteMax r)

-- Split
split :: Ord k => k -> Map k a -> (Map k a, Map k a)
split _ Tip = (Tip, Tip)
split k (Bin _ kx x l r) = case compare k kx of
    LT -> let (lt, gt) = split k l in (lt, link kx x gt r)
    GT -> let (lt, gt) = split k r in (link kx x l lt, gt)
    EQ -> (l, r)

splitLookup :: Ord k => k -> Map k a -> (Map k a, Maybe a, Map k a)
splitLookup _ Tip = (Tip, Nothing, Tip)
splitLookup k (Bin _ kx x l r) = case compare k kx of
    LT -> let (lt, found, gt) = splitLookup k l in (lt, found, link kx x gt r)
    GT -> let (lt, found, gt) = splitLookup k r in (link kx x l lt, found, gt)
    EQ -> (l, Just x, r)

-- Internal helpers
balance :: k -> a -> Map k a -> Map k a -> Map k a
balance k x l r = Bin (size l + size r + 1) k x l r

link :: k -> a -> Map k a -> Map k a -> Map k a
link kx x Tip r = insertMin kx x r
link kx x l Tip = insertMax kx x l
link kx x l r = balance kx x l r

insertMin :: k -> a -> Map k a -> Map k a
insertMin kx x Tip = singleton kx x
insertMin kx x (Bin _ ky y l r) = balance ky y (insertMin kx x l) r

insertMax :: k -> a -> Map k a -> Map k a
insertMax kx x Tip = singleton kx x
insertMax kx x (Bin _ ky y l r) = balance ky y l (insertMax kx x r)

glue :: Map k a -> Map k a -> Map k a
glue Tip r = r
glue l Tip = l
glue l r
    | size l > size r = let (k, x) = findMax l in balance k x (deleteMax l) r
    | otherwise       = let (k, x) = findMin r in balance k x l (deleteMin r)

merge :: Map k a -> Map k a -> Map k a
merge Tip r = r
merge l Tip = l
merge l r = glue l r
