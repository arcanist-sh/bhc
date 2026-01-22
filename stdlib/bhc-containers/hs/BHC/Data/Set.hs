-- |
-- Module      : BHC.Data.Set
-- Description : Ordered sets
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Data.Set (
    Set,
    
    -- * Construction
    empty, singleton, fromList, fromAscList, fromDescList,
    
    -- * Insertion
    insert,
    
    -- * Deletion
    delete,
    
    -- * Query
    member, notMember, lookupLT, lookupGT, lookupLE, lookupGE,
    null, size, isSubsetOf, isProperSubsetOf,
    
    -- * Combine
    union, unions, difference, (\\), intersection,
    disjoint,
    
    -- * Filter
    filter, partition, split, splitMember,
    
    -- * Map
    map, mapMonotonic,
    
    -- * Folds
    foldr, foldl, foldr', foldl',
    
    -- * Min/Max
    lookupMin, lookupMax, findMin, findMax,
    deleteMin, deleteMax, minView, maxView,
    
    -- * Conversion
    elems, toList, toAscList, toDescList,
) where

import BHC.Prelude hiding (map, null, filter, foldr, foldl)
import qualified BHC.Prelude as P

-- | A set of values @a@.
data Set a
    = Tip
    | Bin {-# UNPACK #-} !Int !a !(Set a) !(Set a)
    deriving (Eq, Ord, Show, Read)

instance Ord a => Semigroup (Set a) where
    (<>) = union

instance Ord a => Monoid (Set a) where
    mempty = empty

instance Foldable Set where
    foldr = foldr
    foldl = foldl
    null = null
    length = size
    elem = member
    toList = toList

-- Construction
empty :: Set a
empty = Tip

singleton :: a -> Set a
singleton x = Bin 1 x Tip Tip

fromList :: Ord a => [a] -> Set a
fromList = P.foldl' (flip insert) empty

fromAscList :: Eq a => [a] -> Set a
fromAscList = fromList

fromDescList :: Eq a => [a] -> Set a
fromDescList = fromList

-- Insertion
insert :: Ord a => a -> Set a -> Set a
insert x = go
  where
    go Tip = singleton x
    go (Bin sz y l r) = case compare x y of
        LT -> balance y (go l) r
        GT -> balance y l (go r)
        EQ -> Bin sz x l r

-- Deletion
delete :: Ord a => a -> Set a -> Set a
delete x = go
  where
    go Tip = Tip
    go (Bin _ y l r) = case compare x y of
        LT -> balance y (go l) r
        GT -> balance y l (go r)
        EQ -> glue l r

-- Query
member :: Ord a => a -> Set a -> Bool
member x = go
  where
    go Tip = False
    go (Bin _ y l r) = case compare x y of
        LT -> go l
        GT -> go r
        EQ -> True

notMember :: Ord a => a -> Set a -> Bool
notMember x = not . member x

lookupLT :: Ord a => a -> Set a -> Maybe a
lookupLT = goNothing
  where
    goNothing _ Tip = Nothing
    goNothing x (Bin _ y l r)
        | x <= y = goNothing x l
        | otherwise = goJust x y r
    goJust _ best Tip = Just best
    goJust x best (Bin _ y l r)
        | x <= y = goJust x best l
        | otherwise = goJust x y r

lookupGT :: Ord a => a -> Set a -> Maybe a
lookupGT = goNothing
  where
    goNothing _ Tip = Nothing
    goNothing x (Bin _ y l r)
        | x < y = goJust x y l
        | otherwise = goNothing x r
    goJust _ best Tip = Just best
    goJust x best (Bin _ y l r)
        | x < y = goJust x y l
        | otherwise = goJust x best r

lookupLE :: Ord a => a -> Set a -> Maybe a
lookupLE = goNothing
  where
    goNothing _ Tip = Nothing
    goNothing x (Bin _ y l r) = case compare x y of
        LT -> goNothing x l
        EQ -> Just y
        GT -> goJust x y r
    goJust _ best Tip = Just best
    goJust x best (Bin _ y l r) = case compare x y of
        LT -> goJust x best l
        EQ -> Just y
        GT -> goJust x y r

lookupGE :: Ord a => a -> Set a -> Maybe a
lookupGE = goNothing
  where
    goNothing _ Tip = Nothing
    goNothing x (Bin _ y l r) = case compare x y of
        LT -> goJust x y l
        EQ -> Just y
        GT -> goNothing x r
    goJust _ best Tip = Just best
    goJust x best (Bin _ y l r) = case compare x y of
        LT -> goJust x y l
        EQ -> Just y
        GT -> goJust x best r

null :: Set a -> Bool
null Tip = True
null _   = False

size :: Set a -> Int
size Tip            = 0
size (Bin n _ _ _)  = n

isSubsetOf :: Ord a => Set a -> Set a -> Bool
isSubsetOf t1 t2 = size t1 <= size t2 && P.all (`member` t2) (toList t1)

isProperSubsetOf :: Ord a => Set a -> Set a -> Bool
isProperSubsetOf t1 t2 = size t1 < size t2 && isSubsetOf t1 t2

-- Combine
union :: Ord a => Set a -> Set a -> Set a
union t1 Tip = t1
union Tip t2 = t2
union t1 t2 = P.foldl' (flip insert) t1 (toList t2)

unions :: (Foldable f, Ord a) => f (Set a) -> Set a
unions = P.foldl' union empty

difference :: Ord a => Set a -> Set a -> Set a
difference t1 t2 = P.foldl' (flip delete) t1 (toList t2)

(\\) :: Ord a => Set a -> Set a -> Set a
(\\) = difference
infixl 9 \\

intersection :: Ord a => Set a -> Set a -> Set a
intersection t1 t2 = filter (`member` t2) t1

disjoint :: Ord a => Set a -> Set a -> Bool
disjoint t1 t2 = null (intersection t1 t2)

-- Filter
filter :: (a -> Bool) -> Set a -> Set a
filter _ Tip = Tip
filter p (Bin _ x l r)
    | p x       = link x (filter p l) (filter p r)
    | otherwise = merge (filter p l) (filter p r)

partition :: (a -> Bool) -> Set a -> (Set a, Set a)
partition _ Tip = (Tip, Tip)
partition p (Bin _ x l r)
    | p x       = (link x l1 r1, merge l2 r2)
    | otherwise = (merge l1 r1, link x l2 r2)
  where
    (l1, l2) = partition p l
    (r1, r2) = partition p r

split :: Ord a => a -> Set a -> (Set a, Set a)
split _ Tip = (Tip, Tip)
split x (Bin _ y l r) = case compare x y of
    LT -> let (lt, gt) = split x l in (lt, link y gt r)
    GT -> let (lt, gt) = split x r in (link y l lt, gt)
    EQ -> (l, r)

splitMember :: Ord a => a -> Set a -> (Set a, Bool, Set a)
splitMember _ Tip = (Tip, False, Tip)
splitMember x (Bin _ y l r) = case compare x y of
    LT -> let (lt, found, gt) = splitMember x l in (lt, found, link y gt r)
    GT -> let (lt, found, gt) = splitMember x r in (link y l lt, found, gt)
    EQ -> (l, True, r)

-- Map
map :: Ord b => (a -> b) -> Set a -> Set b
map f = fromList . P.map f . toList

mapMonotonic :: (a -> b) -> Set a -> Set b
mapMonotonic _ Tip = Tip
mapMonotonic f (Bin sz x l r) = Bin sz (f x) (mapMonotonic f l) (mapMonotonic f r)

-- Folds
foldr :: (a -> b -> b) -> b -> Set a -> b
foldr _ z Tip = z
foldr f z (Bin _ x l r) = foldr f (f x (foldr f z r)) l

foldl :: (a -> b -> a) -> a -> Set b -> a
foldl _ z Tip = z
foldl f z (Bin _ x l r) = foldl f (f (foldl f z l) x) r

foldr' :: (a -> b -> b) -> b -> Set a -> b
foldr' f z = go z
  where
    go !z' Tip = z'
    go !z' (Bin _ x l r) = go (f x (go z' r)) l

foldl' :: (a -> b -> a) -> a -> Set b -> a
foldl' f z = go z
  where
    go !z' Tip = z'
    go !z' (Bin _ x l r) = go (f (go z' l) x) r

-- Min/Max
lookupMin :: Set a -> Maybe a
lookupMin Tip = Nothing
lookupMin (Bin _ x Tip _) = Just x
lookupMin (Bin _ _ l _) = lookupMin l

lookupMax :: Set a -> Maybe a
lookupMax Tip = Nothing
lookupMax (Bin _ x _ Tip) = Just x
lookupMax (Bin _ _ _ r) = lookupMax r

findMin :: Set a -> a
findMin s = case lookupMin s of
    Just x  -> x
    Nothing -> error "Set.findMin: empty set"

findMax :: Set a -> a
findMax s = case lookupMax s of
    Just x  -> x
    Nothing -> error "Set.findMax: empty set"

deleteMin :: Set a -> Set a
deleteMin Tip = Tip
deleteMin (Bin _ _ Tip r) = r
deleteMin (Bin _ x l r) = balance x (deleteMin l) r

deleteMax :: Set a -> Set a
deleteMax Tip = Tip
deleteMax (Bin _ _ l Tip) = l
deleteMax (Bin _ x l r) = balance x l (deleteMax r)

minView :: Set a -> Maybe (a, Set a)
minView Tip = Nothing
minView s = Just (findMin s, deleteMin s)

maxView :: Set a -> Maybe (a, Set a)
maxView Tip = Nothing
maxView s = Just (findMax s, deleteMax s)

-- Conversion
elems :: Set a -> [a]
elems = toList

toList :: Set a -> [a]
toList = toAscList

toAscList :: Set a -> [a]
toAscList = foldr (:) []

toDescList :: Set a -> [a]
toDescList = foldl (flip (:)) []

-- Internal
balance :: a -> Set a -> Set a -> Set a
balance x l r = Bin (size l + size r + 1) x l r

link :: a -> Set a -> Set a -> Set a
link x Tip r = insertMin x r
link x l Tip = insertMax x l
link x l r = balance x l r

insertMin :: a -> Set a -> Set a
insertMin x Tip = singleton x
insertMin x (Bin _ y l r) = balance y (insertMin x l) r

insertMax :: a -> Set a -> Set a
insertMax x Tip = singleton x
insertMax x (Bin _ y l r) = balance y l (insertMax x r)

glue :: Set a -> Set a -> Set a
glue Tip r = r
glue l Tip = l
glue l r
    | size l > size r = let m = findMax l in balance m (deleteMax l) r
    | otherwise       = let m = findMin r in balance m l (deleteMin r)

merge :: Set a -> Set a -> Set a
merge Tip r = r
merge l Tip = l
merge l r = glue l r
