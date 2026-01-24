{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveTraversable #-}

-- |
-- Module      : BHC.Data.Sequence
-- Description : Efficient finite sequences based on finger trees
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
--
-- A general-purpose finite sequence type with efficient operations at both ends.
-- Based on finger trees, providing:
--
-- * O(1) access to front and back
-- * O(log n) indexing, splitting, and concatenation
-- * O(n) traversal
--
-- = Usage
--
-- @
-- import qualified BHC.Data.Sequence as Seq
--
-- let s = Seq.fromList [1, 2, 3, 4, 5]
-- Seq.length s  -- 5
-- s Seq.|> 6    -- append to right
-- 0 Seq.<| s    -- prepend to left
-- @

module BHC.Data.Sequence
    ( -- * Sequence type
      Seq

      -- * Construction
    , empty
    , singleton
    , (<|)
    , (|>)
    , (><)
    , fromList
    , fromFunction
    , replicate
    , replicateA
    , replicateM

      -- * Deconstruction
    , null
    , length
    , ViewL(..)
    , viewl
    , ViewR(..)
    , viewr

      -- * Indexing
    , lookup
    , (!?)
    , index
    , adjust
    , adjust'
    , update
    , take
    , drop
    , splitAt
    , insertAt
    , deleteAt

      -- * Predicates
    , elemIndexL
    , elemIndicesL
    , elemIndexR
    , elemIndicesR
    , findIndexL
    , findIndicesL
    , findIndexR
    , findIndicesR

      -- * Transformation
    , reverse
    , intersperse
    , scanl
    , scanl1
    , scanr
    , scanr1

      -- * Sorting
    , sort
    , sortBy
    , sortOn
    , unstableSort
    , unstableSortBy
    , unstableSortOn

      -- * Subsequences
    , tails
    , inits
    , chunksOf

      -- * Sequential searches
    , takeWhileL
    , takeWhileR
    , dropWhileL
    , dropWhileR
    , spanl
    , spanr
    , breakl
    , breakr
    , partition
    , filter

      -- * Zipping
    , zip
    , zipWith
    , zip3
    , zipWith3
    , zip4
    , zipWith4
    , unzip
    , unzipWith

      -- * Folds
    , foldMapWithIndex
    , foldlWithIndex
    , foldrWithIndex

      -- * Traversals
    , traverseWithIndex
    , mapWithIndex
    ) where

import Prelude hiding (null, length, lookup, take, drop, splitAt, reverse,
                       scanl, scanl1, scanr, scanr1, filter, zip, zipWith,
                       zip3, zipWith3, replicate, unzip)
import qualified Prelude

import Control.Applicative (Applicative(..))
import Control.Monad (ap)
import Data.Foldable (Foldable(..), toList)
import Data.Monoid (Monoid(..))
import Data.Semigroup (Semigroup(..))
import GHC.Generics (Generic)

-- ============================================================
-- Core Types
-- ============================================================

-- | A finger tree representing a sequence of elements.
-- Based on Hinze & Paterson's 2-3 finger trees.
data Seq a
    = Empty
    | Single a
    | Deep {-# UNPACK #-} !Int !(Digit a) (Seq (Node a)) !(Digit a)
    deriving (Generic)

-- | A digit contains 1-4 elements (at the edge of the tree).
data Digit a
    = One a
    | Two a a
    | Three a a a
    | Four a a a a
    deriving (Functor, Foldable, Traversable, Generic)

-- | Internal nodes contain 2-3 elements with cached size.
data Node a
    = Node2 {-# UNPACK #-} !Int a a
    | Node3 {-# UNPACK #-} !Int a a a
    deriving (Functor, Foldable, Traversable, Generic)

-- | Left view of a sequence.
data ViewL a
    = EmptyL
    | a :< Seq a
    deriving (Eq, Ord, Show, Functor, Foldable, Traversable, Generic)

-- | Right view of a sequence.
data ViewR a
    = EmptyR
    | Seq a :> a
    deriving (Eq, Ord, Show, Functor, Foldable, Traversable, Generic)

-- ============================================================
-- Instances
-- ============================================================

instance Show a => Show (Seq a) where
    showsPrec p xs = showParen (p > 10) $
        showString "fromList " . shows (toList xs)

instance Eq a => Eq (Seq a) where
    xs == ys = length xs == length ys && toList xs == toList ys

instance Ord a => Ord (Seq a) where
    compare xs ys = compare (toList xs) (toList ys)

instance Semigroup (Seq a) where
    (<>) = (><)

instance Monoid (Seq a) where
    mempty = empty
    mappend = (<>)

instance Functor Seq where
    fmap = mapSeq

instance Foldable Seq where
    foldr f z = go
      where
        go Empty = z
        go (Single x) = f x z
        go (Deep _ pr m sf) = foldr f (foldr (flip (foldr f)) (foldr f z sf) m) pr

    foldl f z = go z
      where
        go !acc Empty = acc
        go !acc (Single x) = f acc x
        go !acc (Deep _ pr m sf) = foldl f (foldl (foldl f) (foldl f acc pr) m) sf

    length = length

instance Traversable Seq where
    traverse f = go
      where
        go Empty = pure Empty
        go (Single x) = Single <$> f x
        go (Deep n pr m sf) = Deep n <$> traverse f pr
                                     <*> traverse (traverse f) m
                                     <*> traverse f sf

instance Applicative Seq where
    pure = singleton
    (<*>) = ap

instance Monad Seq where
    return = pure
    xs >>= f = foldl' (\acc x -> acc >< f x) empty xs

-- ============================================================
-- Construction
-- ============================================================

-- | The empty sequence. O(1)
empty :: Seq a
empty = Empty

-- | A sequence with a single element. O(1)
singleton :: a -> Seq a
singleton = Single

-- | Add an element to the left end. O(1) amortized
infixr 5 <|
(<|) :: a -> Seq a -> Seq a
a <| Empty = Single a
a <| Single b = deep (One a) Empty (One b)
a <| Deep n (Four b c d e) m sf = m `seq` Deep (n + 1) (Two a b) (node3 c d e <| m) sf
a <| Deep n pr m sf = Deep (n + 1) (consDigit a pr) m sf

-- | Add an element to the right end. O(1) amortized
infixl 5 |>
(|>) :: Seq a -> a -> Seq a
Empty |> a = Single a
Single a |> b = deep (One a) Empty (One b)
Deep n pr m (Four a b c d) |> e = m `seq` Deep (n + 1) pr (m |> node3 a b c) (Two d e)
Deep n pr m sf |> a = Deep (n + 1) pr m (snocDigit sf a)

-- | Concatenate two sequences. O(log(min(n,m)))
infixr 5 ><
(><) :: Seq a -> Seq a -> Seq a
Empty >< ys = ys
xs >< Empty = xs
Single x >< ys = x <| ys
xs >< Single y = xs |> y
Deep n1 pr1 m1 sf1 >< Deep n2 pr2 m2 sf2 =
    Deep (n1 + n2) pr1 (addDigits0 m1 sf1 pr2 m2) sf2

-- | Create a sequence from a list. O(n)
fromList :: [a] -> Seq a
fromList = Prelude.foldr (<|) Empty

-- | Create a sequence of given length from a function. O(n)
fromFunction :: Int -> (Int -> a) -> Seq a
fromFunction n f
    | n <= 0    = Empty
    | otherwise = fromList [f i | i <- [0..n-1]]

-- | Create a sequence of n copies of an element. O(n)
replicate :: Int -> a -> Seq a
replicate n x
    | n <= 0    = Empty
    | otherwise = fromList (Prelude.replicate n x)

-- | Applicative replicate
replicateA :: Applicative f => Int -> f a -> f (Seq a)
replicateA n fa
    | n <= 0    = pure Empty
    | otherwise = fromList <$> sequenceA (Prelude.replicate n fa)

-- | Monadic replicate
replicateM :: Monad m => Int -> m a -> m (Seq a)
replicateM n ma
    | n <= 0    = pure Empty
    | otherwise = fromList <$> sequence (Prelude.replicate n ma)

-- ============================================================
-- Deconstruction
-- ============================================================

-- | Is the sequence empty? O(1)
null :: Seq a -> Bool
null Empty = True
null _ = False

-- | The number of elements in the sequence. O(1)
length :: Seq a -> Int
length Empty = 0
length (Single _) = 1
length (Deep n _ _ _) = n

-- | View from the left. O(1) amortized
viewl :: Seq a -> ViewL a
viewl Empty = EmptyL
viewl (Single x) = x :< Empty
viewl (Deep _ (One x) m sf) = x :< pullL m sf
viewl (Deep n pr m sf) = case pr of
    Two a b -> a :< Deep (n - 1) (One b) m sf
    Three a b c -> a :< Deep (n - 1) (Two b c) m sf
    Four a b c d -> a :< Deep (n - 1) (Three b c d) m sf
    One _ -> error "viewl: impossible"

-- | View from the right. O(1) amortized
viewr :: Seq a -> ViewR a
viewr Empty = EmptyR
viewr (Single x) = Empty :> x
viewr (Deep _ pr m (One x)) = pullR pr m :> x
viewr (Deep n pr m sf) = case sf of
    Two a b -> Deep (n - 1) pr m (One a) :> b
    Three a b c -> Deep (n - 1) pr m (Two a b) :> c
    Four a b c d -> Deep (n - 1) pr m (Three a b c) :> d
    One _ -> error "viewr: impossible"

-- ============================================================
-- Indexing
-- ============================================================

-- | Safe indexing. O(log(min(i, n-i)))
lookup :: Int -> Seq a -> Maybe a
lookup i xs
    | i < 0 || i >= length xs = Nothing
    | otherwise = Just (index xs i)

-- | Infix safe indexing
(!?) :: Seq a -> Int -> Maybe a
(!?) = flip lookup

-- | Unsafe indexing. O(log(min(i, n-i)))
index :: Seq a -> Int -> a
index xs i = case lookupTree i xs of
    Just x -> x
    Nothing -> error "Seq.index: index out of bounds"

-- | Adjust element at index. O(log(min(i, n-i)))
adjust :: (a -> a) -> Int -> Seq a -> Seq a
adjust f i xs
    | i < 0 || i >= length xs = xs
    | otherwise = adjustTree f i xs

-- | Strict adjust
adjust' :: (a -> a) -> Int -> Seq a -> Seq a
adjust' f i xs
    | i < 0 || i >= length xs = xs
    | otherwise = adjustTree (\x -> let !y = f x in y) i xs

-- | Update element at index. O(log(min(i, n-i)))
update :: Int -> a -> Seq a -> Seq a
update i x = adjust (const x) i

-- | Take first n elements. O(log(min(n, size-n)))
take :: Int -> Seq a -> Seq a
take i xs
    | i <= 0 = Empty
    | i >= length xs = xs
    | otherwise = fst (splitAt i xs)

-- | Drop first n elements. O(log(min(n, size-n)))
drop :: Int -> Seq a -> Seq a
drop i xs
    | i <= 0 = xs
    | i >= length xs = Empty
    | otherwise = snd (splitAt i xs)

-- | Split at index. O(log(min(i, n-i)))
splitAt :: Int -> Seq a -> (Seq a, Seq a)
splitAt i xs
    | i <= 0 = (Empty, xs)
    | i >= length xs = (xs, Empty)
    | otherwise = splitTreeAt i xs

-- | Insert element at index. O(log(min(i, n-i)))
insertAt :: Int -> a -> Seq a -> Seq a
insertAt i x xs
    | i <= 0 = x <| xs
    | i >= length xs = xs |> x
    | otherwise = let (l, r) = splitAt i xs in l >< singleton x >< r

-- | Delete element at index. O(log(min(i, n-i)))
deleteAt :: Int -> Seq a -> Seq a
deleteAt i xs
    | i < 0 || i >= length xs = xs
    | otherwise = let (l, r) = splitAt i xs
                  in case viewl r of
                      EmptyL -> l
                      _ :< r' -> l >< r'

-- ============================================================
-- Searching
-- ============================================================

-- | Index of first occurrence from left
elemIndexL :: Eq a => a -> Seq a -> Maybe Int
elemIndexL x = findIndexL (== x)

-- | All indices of element from left
elemIndicesL :: Eq a => a -> Seq a -> [Int]
elemIndicesL x = findIndicesL (== x)

-- | Index of first occurrence from right
elemIndexR :: Eq a => a -> Seq a -> Maybe Int
elemIndexR x = findIndexR (== x)

-- | All indices of element from right
elemIndicesR :: Eq a => a -> Seq a -> [Int]
elemIndicesR x = findIndicesR (== x)

-- | Find first index satisfying predicate from left
findIndexL :: (a -> Bool) -> Seq a -> Maybe Int
findIndexL p xs = go 0 xs
  where
    go _ Empty = Nothing
    go !i (Single x)
        | p x = Just i
        | otherwise = Nothing
    go !i (Deep _ pr m sf) =
        case findInDigit p i pr of
            Just j -> Just j
            Nothing ->
                let i' = i + sizeDigit pr
                in case findInMiddle p i' m of
                    Just j -> Just j
                    Nothing -> findInDigit p (i' + sizeMiddle m) sf

-- | Find all indices satisfying predicate from left
findIndicesL :: (a -> Bool) -> Seq a -> [Int]
findIndicesL p xs = [i | (i, x) <- Prelude.zip [0..] (toList xs), p x]

-- | Find first index satisfying predicate from right
findIndexR :: (a -> Bool) -> Seq a -> Maybe Int
findIndexR p xs = go (length xs - 1) xs
  where
    go _ Empty = Nothing
    go !i (Single x)
        | p x = Just i
        | otherwise = Nothing
    go !i (Deep _ pr m sf) =
        case findInDigitR p i sf of
            Just j -> Just j
            Nothing ->
                let i' = i - sizeDigit sf
                in case findInMiddleR p i' m of
                    Just j -> Just j
                    Nothing -> findInDigitR p (i' - sizeMiddle m) pr

-- | Find all indices satisfying predicate from right
findIndicesR :: (a -> Bool) -> Seq a -> [Int]
findIndicesR p xs = Prelude.reverse (findIndicesL p xs)

-- ============================================================
-- Transformation
-- ============================================================

-- | Reverse a sequence. O(n)
reverse :: Seq a -> Seq a
reverse = foldl (flip (<|)) Empty

-- | Intersperse an element between elements. O(n)
intersperse :: a -> Seq a -> Seq a
intersperse sep xs = case viewl xs of
    EmptyL -> Empty
    h :< t -> h <| foldl (\acc x -> acc |> sep |> x) Empty t

-- | Left scan. O(n)
scanl :: (a -> b -> a) -> a -> Seq b -> Seq a
scanl f z xs = z <| case viewl xs of
    EmptyL -> Empty
    h :< t -> scanl f (f z h) t

-- | Left scan without starting value. O(n)
scanl1 :: (a -> a -> a) -> Seq a -> Seq a
scanl1 f xs = case viewl xs of
    EmptyL -> Empty
    h :< t -> scanl f h t

-- | Right scan. O(n)
scanr :: (a -> b -> b) -> b -> Seq a -> Seq b
scanr f z xs = case viewr xs of
    EmptyR -> singleton z
    t :> h -> let ys = scanr f z t
              in case viewl ys of
                  EmptyL -> singleton (f h z)
                  y :< _ -> f h y <| ys

-- | Right scan without starting value. O(n)
scanr1 :: (a -> a -> a) -> Seq a -> Seq a
scanr1 f xs = case viewr xs of
    EmptyR -> Empty
    t :> h -> scanr f h t

-- ============================================================
-- Sorting
-- ============================================================

-- | Stable sort. O(n log n)
sort :: Ord a => Seq a -> Seq a
sort = sortBy compare

-- | Stable sort with custom comparison. O(n log n)
sortBy :: (a -> a -> Ordering) -> Seq a -> Seq a
sortBy cmp xs = fromList (mergeSort cmp (toList xs))

-- | Stable sort on projection. O(n log n)
sortOn :: Ord b => (a -> b) -> Seq a -> Seq a
sortOn f = sortBy (\x y -> compare (f x) (f y))

-- | Unstable sort (potentially faster). O(n log n)
unstableSort :: Ord a => Seq a -> Seq a
unstableSort = sortBy compare

-- | Unstable sort with custom comparison. O(n log n)
unstableSortBy :: (a -> a -> Ordering) -> Seq a -> Seq a
unstableSortBy = sortBy

-- | Unstable sort on projection. O(n log n)
unstableSortOn :: Ord b => (a -> b) -> Seq a -> Seq a
unstableSortOn = sortOn

-- ============================================================
-- Subsequences
-- ============================================================

-- | All suffixes. O(n)
tails :: Seq a -> Seq (Seq a)
tails xs = xs <| case viewl xs of
    EmptyL -> Empty
    _ :< t -> tails t

-- | All prefixes. O(n)
inits :: Seq a -> Seq (Seq a)
inits xs = Empty <| case viewl xs of
    EmptyL -> Empty
    h :< t -> fmap (h <|) (inits t)

-- | Split into chunks of given size. O(n)
chunksOf :: Int -> Seq a -> Seq (Seq a)
chunksOf k xs
    | k <= 0 = Empty
    | null xs = Empty
    | otherwise = let (chunk, rest) = splitAt k xs
                  in chunk <| chunksOf k rest

-- ============================================================
-- Filtering
-- ============================================================

-- | Take elements from left while predicate holds. O(i)
takeWhileL :: (a -> Bool) -> Seq a -> Seq a
takeWhileL p xs = case viewl xs of
    EmptyL -> Empty
    h :< t
        | p h -> h <| takeWhileL p t
        | otherwise -> Empty

-- | Take elements from right while predicate holds. O(i)
takeWhileR :: (a -> Bool) -> Seq a -> Seq a
takeWhileR p xs = case viewr xs of
    EmptyR -> Empty
    t :> h
        | p h -> takeWhileR p t |> h
        | otherwise -> Empty

-- | Drop elements from left while predicate holds. O(i)
dropWhileL :: (a -> Bool) -> Seq a -> Seq a
dropWhileL p xs = case viewl xs of
    EmptyL -> Empty
    h :< t
        | p h -> dropWhileL p t
        | otherwise -> xs

-- | Drop elements from right while predicate holds. O(i)
dropWhileR :: (a -> Bool) -> Seq a -> Seq a
dropWhileR p xs = case viewr xs of
    EmptyR -> Empty
    t :> h
        | p h -> dropWhileR p t
        | otherwise -> xs

-- | Span from left. O(i)
spanl :: (a -> Bool) -> Seq a -> (Seq a, Seq a)
spanl p xs = (takeWhileL p xs, dropWhileL p xs)

-- | Span from right. O(i)
spanr :: (a -> Bool) -> Seq a -> (Seq a, Seq a)
spanr p xs = (dropWhileR p xs, takeWhileR p xs)

-- | Break from left. O(i)
breakl :: (a -> Bool) -> Seq a -> (Seq a, Seq a)
breakl p = spanl (not . p)

-- | Break from right. O(i)
breakr :: (a -> Bool) -> Seq a -> (Seq a, Seq a)
breakr p = spanr (not . p)

-- | Partition by predicate. O(n)
partition :: (a -> Bool) -> Seq a -> (Seq a, Seq a)
partition p xs = foldl go (Empty, Empty) xs
  where
    go (!ts, !fs) x
        | p x = (ts |> x, fs)
        | otherwise = (ts, fs |> x)

-- | Filter elements. O(n)
filter :: (a -> Bool) -> Seq a -> Seq a
filter p = foldl (\acc x -> if p x then acc |> x else acc) Empty

-- ============================================================
-- Zipping
-- ============================================================

-- | Zip two sequences. O(min(n, m))
zip :: Seq a -> Seq b -> Seq (a, b)
zip = zipWith (,)

-- | Zip with function. O(min(n, m))
zipWith :: (a -> b -> c) -> Seq a -> Seq b -> Seq c
zipWith f xs ys = fromList (Prelude.zipWith f (toList xs) (toList ys))

-- | Zip three sequences. O(min(n, m, o))
zip3 :: Seq a -> Seq b -> Seq c -> Seq (a, b, c)
zip3 = zipWith3 (,,)

-- | Zip three with function. O(min(n, m, o))
zipWith3 :: (a -> b -> c -> d) -> Seq a -> Seq b -> Seq c -> Seq d
zipWith3 f xs ys zs = fromList (Prelude.zipWith3 f (toList xs) (toList ys) (toList zs))

-- | Zip four sequences. O(min lengths)
zip4 :: Seq a -> Seq b -> Seq c -> Seq d -> Seq (a, b, c, d)
zip4 = zipWith4 (,,,)

-- | Zip four with function. O(min lengths)
zipWith4 :: (a -> b -> c -> d -> e) -> Seq a -> Seq b -> Seq c -> Seq d -> Seq e
zipWith4 f as bs cs ds = fromList (Prelude.zipWith4 f (toList as) (toList bs) (toList cs) (toList ds))

-- | Unzip a sequence of pairs. O(n)
unzip :: Seq (a, b) -> (Seq a, Seq b)
unzip xs = (fmap fst xs, fmap snd xs)

-- | Unzip with function. O(n)
unzipWith :: (a -> (b, c)) -> Seq a -> (Seq b, Seq c)
unzipWith f xs = unzip (fmap f xs)

-- ============================================================
-- Indexed operations
-- ============================================================

-- | Fold with index. O(n)
foldMapWithIndex :: Monoid m => (Int -> a -> m) -> Seq a -> m
foldMapWithIndex f xs = go 0 xs
  where
    go _ Empty = mempty
    go !i (Single x) = f i x
    go !i (Deep _ pr m sf) =
        foldMapDigitWithIndex f i pr <>
        foldMapMiddleWithIndex f (i + sizeDigit pr) m <>
        foldMapDigitWithIndex f (i + sizeDigit pr + sizeMiddle m) sf

-- | Left fold with index. O(n)
foldlWithIndex :: (b -> Int -> a -> b) -> b -> Seq a -> b
foldlWithIndex f z xs = go 0 z xs
  where
    go _ !acc Empty = acc
    go !i !acc (Single x) = f acc i x
    go !i !acc (Deep _ pr m sf) =
        let !acc1 = foldlDigitWithIndex f acc i pr
            !i1 = i + sizeDigit pr
            !acc2 = foldlMiddleWithIndex f acc1 i1 m
            !i2 = i1 + sizeMiddle m
        in foldlDigitWithIndex f acc2 i2 sf

-- | Right fold with index. O(n)
foldrWithIndex :: (Int -> a -> b -> b) -> b -> Seq a -> b
foldrWithIndex f z xs = go 0 xs z
  where
    go _ Empty !acc = acc
    go !i (Single x) !acc = f i x acc
    go !i (Deep _ pr m sf) !acc =
        let !i2 = i + sizeDigit pr + sizeMiddle m
            !acc1 = foldrDigitWithIndex f acc i2 sf
            !i1 = i + sizeDigit pr
            !acc2 = foldrMiddleWithIndex f acc1 i1 m
        in foldrDigitWithIndex f acc2 i pr

-- | Traverse with index. O(n)
traverseWithIndex :: Applicative f => (Int -> a -> f b) -> Seq a -> f (Seq b)
traverseWithIndex f xs = go 0 xs
  where
    go _ Empty = pure Empty
    go !i (Single x) = Single <$> f i x
    go !i (Deep n pr m sf) =
        Deep n <$> traverseDigitWithIndex f i pr
               <*> traverseMiddleWithIndex f (i + sizeDigit pr) m
               <*> traverseDigitWithIndex f (i + sizeDigit pr + sizeMiddle m) sf

-- | Map with index. O(n)
mapWithIndex :: (Int -> a -> b) -> Seq a -> Seq b
mapWithIndex f xs = go 0 xs
  where
    go _ Empty = Empty
    go !i (Single x) = Single (f i x)
    go !i (Deep n pr m sf) =
        Deep n (mapDigitWithIndex f i pr)
               (mapMiddleWithIndex f (i + sizeDigit pr) m)
               (mapDigitWithIndex f (i + sizeDigit pr + sizeMiddle m) sf)

-- ============================================================
-- Internal: Node operations
-- ============================================================

node2 :: a -> a -> Node a
node2 a b = Node2 2 a b

node3 :: a -> a -> a -> Node a
node3 a b c = Node3 3 a b c

sizeNode :: Node a -> Int
sizeNode (Node2 s _ _) = s
sizeNode (Node3 s _ _ _) = s

nodeToDigit :: Node a -> Digit a
nodeToDigit (Node2 _ a b) = Two a b
nodeToDigit (Node3 _ a b c) = Three a b c

-- ============================================================
-- Internal: Digit operations
-- ============================================================

consDigit :: a -> Digit a -> Digit a
consDigit a (One b) = Two a b
consDigit a (Two b c) = Three a b c
consDigit a (Three b c d) = Four a b c d
consDigit _ (Four _ _ _ _) = error "consDigit: full"

snocDigit :: Digit a -> a -> Digit a
snocDigit (One a) b = Two a b
snocDigit (Two a b) c = Three a b c
snocDigit (Three a b c) d = Four a b c d
snocDigit (Four _ _ _ _) _ = error "snocDigit: full"

sizeDigit :: Digit a -> Int
sizeDigit (One _) = 1
sizeDigit (Two _ _) = 2
sizeDigit (Three _ _ _) = 3
sizeDigit (Four _ _ _ _) = 4

digitToList :: Digit a -> [a]
digitToList (One a) = [a]
digitToList (Two a b) = [a, b]
digitToList (Three a b c) = [a, b, c]
digitToList (Four a b c d) = [a, b, c, d]

-- ============================================================
-- Internal: Deep construction
-- ============================================================

deep :: Digit a -> Seq (Node a) -> Digit a -> Seq a
deep pr m sf = Deep (sizeDigit pr + sizeMiddle m + sizeDigit sf) pr m sf

sizeMiddle :: Seq (Node a) -> Int
sizeMiddle Empty = 0
sizeMiddle (Single n) = sizeNode n
sizeMiddle (Deep n _ _ _) = n

pullL :: Seq (Node a) -> Digit a -> Seq a
pullL m sf = case viewl m of
    EmptyL -> digitToSeq sf
    n :< m' -> Deep (sizeMiddle m + sizeDigit sf) (nodeToDigit n) m' sf

pullR :: Digit a -> Seq (Node a) -> Seq a
pullR pr m = case viewr m of
    EmptyR -> digitToSeq pr
    m' :> n -> Deep (sizeDigit pr + sizeMiddle m) pr m' (nodeToDigit n)

digitToSeq :: Digit a -> Seq a
digitToSeq (One a) = Single a
digitToSeq (Two a b) = deep (One a) Empty (One b)
digitToSeq (Three a b c) = deep (Two a b) Empty (One c)
digitToSeq (Four a b c d) = deep (Two a b) Empty (Two c d)

-- ============================================================
-- Internal: Concatenation helpers
-- ============================================================

addDigits0 :: Seq (Node a) -> Digit a -> Digit a -> Seq (Node a) -> Seq (Node a)
addDigits0 m1 sf1 pr2 m2 =
    appendTree1 m1 (nodes (digitToList sf1 ++ digitToList pr2)) m2

appendTree1 :: Seq (Node a) -> Node a -> Seq (Node a) -> Seq (Node a)
appendTree1 Empty a xs = a <| xs
appendTree1 xs a Empty = xs |> a
appendTree1 (Single x) a xs = x <| a <| xs
appendTree1 xs a (Single x) = xs |> a |> x
appendTree1 (Deep n1 pr1 m1 sf1) a (Deep n2 pr2 m2 sf2) =
    Deep (n1 + sizeNode a + n2) pr1 (addDigits1 m1 sf1 a pr2 m2) sf2

addDigits1 :: Seq (Node (Node a)) -> Digit (Node a) -> Node a -> Digit (Node a) -> Seq (Node (Node a)) -> Seq (Node (Node a))
addDigits1 m1 sf1 a pr2 m2 =
    appendTree1 m1 (nodes (digitToList sf1 ++ [a] ++ digitToList pr2)) m2

nodes :: [a] -> Node a
nodes [a, b] = node2 a b
nodes [a, b, c] = node3 a b c
nodes [a, b, c, d] = error "nodes: too many"  -- This shouldn't happen with proper digit handling
nodes _ = error "nodes: wrong count"

-- ============================================================
-- Internal: Lookup/adjust helpers
-- ============================================================

lookupTree :: Int -> Seq a -> Maybe a
lookupTree _ Empty = Nothing
lookupTree 0 (Single x) = Just x
lookupTree _ (Single _) = Nothing
lookupTree i (Deep _ pr m sf)
    | i < spr = lookupDigit i pr
    | i < spr + sm = lookupMiddle (i - spr) m
    | otherwise = lookupDigit (i - spr - sm) sf
  where
    spr = sizeDigit pr
    sm = sizeMiddle m

lookupDigit :: Int -> Digit a -> Maybe a
lookupDigit 0 (One a) = Just a
lookupDigit 0 (Two a _) = Just a
lookupDigit 1 (Two _ b) = Just b
lookupDigit 0 (Three a _ _) = Just a
lookupDigit 1 (Three _ b _) = Just b
lookupDigit 2 (Three _ _ c) = Just c
lookupDigit 0 (Four a _ _ _) = Just a
lookupDigit 1 (Four _ b _ _) = Just b
lookupDigit 2 (Four _ _ c _) = Just c
lookupDigit 3 (Four _ _ _ d) = Just d
lookupDigit _ _ = Nothing

lookupMiddle :: Int -> Seq (Node a) -> Maybe a
lookupMiddle _ Empty = Nothing
lookupMiddle i (Single n) = lookupNode i n
lookupMiddle i (Deep _ pr m sf)
    | i < spr = lookupDigitN i pr
    | i < spr + sm = lookupMiddle (i - spr) m
    | otherwise = lookupDigitN (i - spr - sm) sf
  where
    spr = sizeDigitN pr
    sm = sizeMiddle m

lookupNode :: Int -> Node a -> Maybe a
lookupNode 0 (Node2 _ a _) = Just a
lookupNode 1 (Node2 _ _ b) = Just b
lookupNode 0 (Node3 _ a _ _) = Just a
lookupNode 1 (Node3 _ _ b _) = Just b
lookupNode 2 (Node3 _ _ _ c) = Just c
lookupNode _ _ = Nothing

lookupDigitN :: Int -> Digit (Node a) -> Maybe a
lookupDigitN i (One n)
    | i < sizeNode n = lookupNode i n
    | otherwise = Nothing
lookupDigitN i (Two n1 n2)
    | i < s1 = lookupNode i n1
    | otherwise = lookupNode (i - s1) n2
  where s1 = sizeNode n1
lookupDigitN i (Three n1 n2 n3)
    | i < s1 = lookupNode i n1
    | i < s1 + s2 = lookupNode (i - s1) n2
    | otherwise = lookupNode (i - s1 - s2) n3
  where
    s1 = sizeNode n1
    s2 = sizeNode n2
lookupDigitN i (Four n1 n2 n3 n4)
    | i < s1 = lookupNode i n1
    | i < s1 + s2 = lookupNode (i - s1) n2
    | i < s1 + s2 + s3 = lookupNode (i - s1 - s2) n3
    | otherwise = lookupNode (i - s1 - s2 - s3) n4
  where
    s1 = sizeNode n1
    s2 = sizeNode n2
    s3 = sizeNode n3

sizeDigitN :: Digit (Node a) -> Int
sizeDigitN (One n) = sizeNode n
sizeDigitN (Two n1 n2) = sizeNode n1 + sizeNode n2
sizeDigitN (Three n1 n2 n3) = sizeNode n1 + sizeNode n2 + sizeNode n3
sizeDigitN (Four n1 n2 n3 n4) = sizeNode n1 + sizeNode n2 + sizeNode n3 + sizeNode n4

adjustTree :: (a -> a) -> Int -> Seq a -> Seq a
adjustTree _ _ Empty = Empty
adjustTree f 0 (Single x) = Single (f x)
adjustTree _ _ (Single x) = Single x
adjustTree f i (Deep n pr m sf)
    | i < spr = Deep n (adjustDigit f i pr) m sf
    | i < spr + sm = Deep n pr (adjustMiddle f (i - spr) m) sf
    | otherwise = Deep n pr m (adjustDigit f (i - spr - sm) sf)
  where
    spr = sizeDigit pr
    sm = sizeMiddle m

adjustDigit :: (a -> a) -> Int -> Digit a -> Digit a
adjustDigit f 0 (One a) = One (f a)
adjustDigit f 0 (Two a b) = Two (f a) b
adjustDigit f 1 (Two a b) = Two a (f b)
adjustDigit f 0 (Three a b c) = Three (f a) b c
adjustDigit f 1 (Three a b c) = Three a (f b) c
adjustDigit f 2 (Three a b c) = Three a b (f c)
adjustDigit f 0 (Four a b c d) = Four (f a) b c d
adjustDigit f 1 (Four a b c d) = Four a (f b) c d
adjustDigit f 2 (Four a b c d) = Four a b (f c) d
adjustDigit f 3 (Four a b c d) = Four a b c (f d)
adjustDigit _ _ d = d

adjustMiddle :: (a -> a) -> Int -> Seq (Node a) -> Seq (Node a)
adjustMiddle _ _ Empty = Empty
adjustMiddle f i (Single n) = Single (adjustNode f i n)
adjustMiddle f i (Deep n pr m sf)
    | i < spr = Deep n (adjustDigitN f i pr) m sf
    | i < spr + sm = Deep n pr (adjustMiddle f (i - spr) m) sf
    | otherwise = Deep n pr m (adjustDigitN f (i - spr - sm) sf)
  where
    spr = sizeDigitN pr
    sm = sizeMiddle m

adjustNode :: (a -> a) -> Int -> Node a -> Node a
adjustNode f 0 (Node2 s a b) = Node2 s (f a) b
adjustNode f 1 (Node2 s a b) = Node2 s a (f b)
adjustNode f 0 (Node3 s a b c) = Node3 s (f a) b c
adjustNode f 1 (Node3 s a b c) = Node3 s a (f b) c
adjustNode f 2 (Node3 s a b c) = Node3 s a b (f c)
adjustNode _ _ n = n

adjustDigitN :: (a -> a) -> Int -> Digit (Node a) -> Digit (Node a)
adjustDigitN f i (One n)
    | i < sizeNode n = One (adjustNode f i n)
    | otherwise = One n
adjustDigitN f i (Two n1 n2)
    | i < s1 = Two (adjustNode f i n1) n2
    | otherwise = Two n1 (adjustNode f (i - s1) n2)
  where s1 = sizeNode n1
adjustDigitN f i (Three n1 n2 n3)
    | i < s1 = Three (adjustNode f i n1) n2 n3
    | i < s1 + s2 = Three n1 (adjustNode f (i - s1) n2) n3
    | otherwise = Three n1 n2 (adjustNode f (i - s1 - s2) n3)
  where
    s1 = sizeNode n1
    s2 = sizeNode n2
adjustDigitN f i (Four n1 n2 n3 n4)
    | i < s1 = Four (adjustNode f i n1) n2 n3 n4
    | i < s1 + s2 = Four n1 (adjustNode f (i - s1) n2) n3 n4
    | i < s1 + s2 + s3 = Four n1 n2 (adjustNode f (i - s1 - s2) n3) n4
    | otherwise = Four n1 n2 n3 (adjustNode f (i - s1 - s2 - s3) n4)
  where
    s1 = sizeNode n1
    s2 = sizeNode n2
    s3 = sizeNode n3

-- ============================================================
-- Internal: Split helpers
-- ============================================================

splitTreeAt :: Int -> Seq a -> (Seq a, Seq a)
splitTreeAt _ Empty = (Empty, Empty)
splitTreeAt i (Single x)
    | i <= 0 = (Empty, Single x)
    | otherwise = (Single x, Empty)
splitTreeAt i (Deep _ pr m sf)
    | i < spr = let (l, r) = splitDigitAt i pr
                in (digitToSeq' l, maybeDeep r m sf)
    | i < spr + sm = let (ml, x, mr) = splitMiddleAt (i - spr) m
                         (l, r) = splitNode (i - spr - sizeMiddle ml) x
                     in (maybeDeep' pr ml l, maybeDeep r mr sf)
    | otherwise = let (l, r) = splitDigitAt (i - spr - sm) sf
                  in (maybeDeep' pr m l, digitToSeq' r)
  where
    spr = sizeDigit pr
    sm = sizeMiddle m

splitDigitAt :: Int -> Digit a -> ([a], [a])
splitDigitAt i d = Prelude.splitAt i (digitToList d)

splitMiddleAt :: Int -> Seq (Node a) -> (Seq (Node a), Node a, Seq (Node a))
splitMiddleAt _ Empty = error "splitMiddleAt: empty"
splitMiddleAt _ (Single n) = (Empty, n, Empty)
splitMiddleAt i (Deep _ pr m sf)
    | i < spr = let (l, x, r) = splitDigitNAt i pr
                in (digitToSeqN l, x, maybeDeepN r m sf)
    | i < spr + sm = let (ml, x, mr) = splitMiddleAt (i - spr) m
                         (l, n, r) = splitNodeN (i - spr - sizeMiddle ml) x
                     in (maybeDeepN' pr ml l, n, maybeDeepN r mr sf)
    | otherwise = let (l, x, r) = splitDigitNAt (i - spr - sm) sf
                  in (maybeDeepN' pr m l, x, digitToSeqN r)
  where
    spr = sizeDigitN pr
    sm = sizeMiddle m

splitDigitNAt :: Int -> Digit (Node a) -> ([Node a], Node a, [Node a])
splitDigitNAt i d = go i (digitToList d)
  where
    go _ [] = error "splitDigitNAt: empty"
    go j (n:ns)
        | j < sizeNode n = ([], n, ns)
        | otherwise = let (l, x, r) = go (j - sizeNode n) ns
                      in (n:l, x, r)

splitNode :: Int -> Node a -> ([a], [a])
splitNode i (Node2 _ a b)
    | i <= 0 = ([], [a, b])
    | i == 1 = ([a], [b])
    | otherwise = ([a, b], [])
splitNode i (Node3 _ a b c)
    | i <= 0 = ([], [a, b, c])
    | i == 1 = ([a], [b, c])
    | i == 2 = ([a, b], [c])
    | otherwise = ([a, b, c], [])

splitNodeN :: Int -> Node (Node a) -> ([Node a], Node a, [Node a])
splitNodeN i (Node2 _ n1 n2)
    | i < sizeNode n1 = ([], n1, [n2])
    | otherwise = ([n1], n2, [])
splitNodeN i (Node3 _ n1 n2 n3)
    | i < s1 = ([], n1, [n2, n3])
    | i < s1 + s2 = ([n1], n2, [n3])
    | otherwise = ([n1, n2], n3, [])
  where
    s1 = sizeNode n1
    s2 = sizeNode n2

digitToSeq' :: [a] -> Seq a
digitToSeq' [] = Empty
digitToSeq' [a] = Single a
digitToSeq' [a, b] = deep (One a) Empty (One b)
digitToSeq' [a, b, c] = deep (Two a b) Empty (One c)
digitToSeq' [a, b, c, d] = deep (Two a b) Empty (Two c d)
digitToSeq' _ = error "digitToSeq': too many"

digitToSeqN :: [Node a] -> Seq (Node a)
digitToSeqN [] = Empty
digitToSeqN [n] = Single n
digitToSeqN [n1, n2] = deep (One n1) Empty (One n2)
digitToSeqN [n1, n2, n3] = deep (Two n1 n2) Empty (One n3)
digitToSeqN [n1, n2, n3, n4] = deep (Two n1 n2) Empty (Two n3 n4)
digitToSeqN _ = error "digitToSeqN: too many"

maybeDeep :: [a] -> Seq (Node a) -> Digit a -> Seq a
maybeDeep [] m sf = pullL m sf
maybeDeep [a] m sf = Deep (1 + sizeMiddle m + sizeDigit sf) (One a) m sf
maybeDeep [a, b] m sf = Deep (2 + sizeMiddle m + sizeDigit sf) (Two a b) m sf
maybeDeep [a, b, c] m sf = Deep (3 + sizeMiddle m + sizeDigit sf) (Three a b c) m sf
maybeDeep [a, b, c, d] m sf = Deep (4 + sizeMiddle m + sizeDigit sf) (Four a b c d) m sf
maybeDeep _ _ _ = error "maybeDeep: too many"

maybeDeep' :: Digit a -> Seq (Node a) -> [a] -> Seq a
maybeDeep' pr m [] = pullR pr m
maybeDeep' pr m [a] = Deep (sizeDigit pr + sizeMiddle m + 1) pr m (One a)
maybeDeep' pr m [a, b] = Deep (sizeDigit pr + sizeMiddle m + 2) pr m (Two a b)
maybeDeep' pr m [a, b, c] = Deep (sizeDigit pr + sizeMiddle m + 3) pr m (Three a b c)
maybeDeep' pr m [a, b, c, d] = Deep (sizeDigit pr + sizeMiddle m + 4) pr m (Four a b c d)
maybeDeep' _ _ _ = error "maybeDeep': too many"

maybeDeepN :: [Node a] -> Seq (Node (Node a)) -> Digit (Node a) -> Seq (Node a)
maybeDeepN [] m sf = pullL m sf
maybeDeepN [n] m sf = Deep (sizeNode n + sizeMiddle m + sizeDigitN sf) (One n) m sf
maybeDeepN [n1, n2] m sf = Deep (sizeNode n1 + sizeNode n2 + sizeMiddle m + sizeDigitN sf) (Two n1 n2) m sf
maybeDeepN [n1, n2, n3] m sf = Deep (sizeNode n1 + sizeNode n2 + sizeNode n3 + sizeMiddle m + sizeDigitN sf) (Three n1 n2 n3) m sf
maybeDeepN [n1, n2, n3, n4] m sf = Deep (sizeNode n1 + sizeNode n2 + sizeNode n3 + sizeNode n4 + sizeMiddle m + sizeDigitN sf) (Four n1 n2 n3 n4) m sf
maybeDeepN _ _ _ = error "maybeDeepN: too many"

maybeDeepN' :: Digit (Node a) -> Seq (Node (Node a)) -> [Node a] -> Seq (Node a)
maybeDeepN' pr m [] = pullR pr m
maybeDeepN' pr m [n] = Deep (sizeDigitN pr + sizeMiddle m + sizeNode n) pr m (One n)
maybeDeepN' pr m [n1, n2] = Deep (sizeDigitN pr + sizeMiddle m + sizeNode n1 + sizeNode n2) pr m (Two n1 n2)
maybeDeepN' pr m [n1, n2, n3] = Deep (sizeDigitN pr + sizeMiddle m + sizeNode n1 + sizeNode n2 + sizeNode n3) pr m (Three n1 n2 n3)
maybeDeepN' pr m [n1, n2, n3, n4] = Deep (sizeDigitN pr + sizeMiddle m + sizeNode n1 + sizeNode n2 + sizeNode n3 + sizeNode n4) pr m (Four n1 n2 n3 n4)
maybeDeepN' _ _ _ = error "maybeDeepN': too many"

-- ============================================================
-- Internal: Map helper
-- ============================================================

mapSeq :: (a -> b) -> Seq a -> Seq b
mapSeq _ Empty = Empty
mapSeq f (Single x) = Single (f x)
mapSeq f (Deep n pr m sf) = Deep n (fmap f pr) (mapSeq (fmap f) m) (fmap f sf)

-- ============================================================
-- Internal: Search helpers
-- ============================================================

findInDigit :: (a -> Bool) -> Int -> Digit a -> Maybe Int
findInDigit p i (One a)
    | p a = Just i
    | otherwise = Nothing
findInDigit p i (Two a b)
    | p a = Just i
    | p b = Just (i + 1)
    | otherwise = Nothing
findInDigit p i (Three a b c)
    | p a = Just i
    | p b = Just (i + 1)
    | p c = Just (i + 2)
    | otherwise = Nothing
findInDigit p i (Four a b c d)
    | p a = Just i
    | p b = Just (i + 1)
    | p c = Just (i + 2)
    | p d = Just (i + 3)
    | otherwise = Nothing

findInDigitR :: (a -> Bool) -> Int -> Digit a -> Maybe Int
findInDigitR p i (One a)
    | p a = Just i
    | otherwise = Nothing
findInDigitR p i (Two a b)
    | p b = Just i
    | p a = Just (i - 1)
    | otherwise = Nothing
findInDigitR p i (Three a b c)
    | p c = Just i
    | p b = Just (i - 1)
    | p a = Just (i - 2)
    | otherwise = Nothing
findInDigitR p i (Four a b c d)
    | p d = Just i
    | p c = Just (i - 1)
    | p b = Just (i - 2)
    | p a = Just (i - 3)
    | otherwise = Nothing

findInMiddle :: (a -> Bool) -> Int -> Seq (Node a) -> Maybe Int
findInMiddle _ _ Empty = Nothing
findInMiddle p i (Single n) = findInNode p i n
findInMiddle p i (Deep _ pr m sf) =
    case findInDigitN p i pr of
        Just j -> Just j
        Nothing ->
            let i' = i + sizeDigitN pr
            in case findInMiddle p i' m of
                Just j -> Just j
                Nothing -> findInDigitN p (i' + sizeMiddle m) sf

findInMiddleR :: (a -> Bool) -> Int -> Seq (Node a) -> Maybe Int
findInMiddleR _ _ Empty = Nothing
findInMiddleR p i (Single n) = findInNodeR p i n
findInMiddleR p i (Deep _ pr m sf) =
    case findInDigitNR p i sf of
        Just j -> Just j
        Nothing ->
            let i' = i - sizeDigitN sf
            in case findInMiddleR p i' m of
                Just j -> Just j
                Nothing -> findInDigitNR p (i' - sizeMiddle m) pr

findInNode :: (a -> Bool) -> Int -> Node a -> Maybe Int
findInNode p i (Node2 _ a b)
    | p a = Just i
    | p b = Just (i + 1)
    | otherwise = Nothing
findInNode p i (Node3 _ a b c)
    | p a = Just i
    | p b = Just (i + 1)
    | p c = Just (i + 2)
    | otherwise = Nothing

findInNodeR :: (a -> Bool) -> Int -> Node a -> Maybe Int
findInNodeR p i (Node2 _ a b)
    | p b = Just i
    | p a = Just (i - 1)
    | otherwise = Nothing
findInNodeR p i (Node3 _ a b c)
    | p c = Just i
    | p b = Just (i - 1)
    | p a = Just (i - 2)
    | otherwise = Nothing

findInDigitN :: (a -> Bool) -> Int -> Digit (Node a) -> Maybe Int
findInDigitN p i (One n) = findInNode p i n
findInDigitN p i (Two n1 n2) =
    case findInNode p i n1 of
        Just j -> Just j
        Nothing -> findInNode p (i + sizeNode n1) n2
findInDigitN p i (Three n1 n2 n3) =
    case findInNode p i n1 of
        Just j -> Just j
        Nothing -> case findInNode p (i + sizeNode n1) n2 of
            Just j -> Just j
            Nothing -> findInNode p (i + sizeNode n1 + sizeNode n2) n3
findInDigitN p i (Four n1 n2 n3 n4) =
    case findInNode p i n1 of
        Just j -> Just j
        Nothing -> case findInNode p (i + sizeNode n1) n2 of
            Just j -> Just j
            Nothing -> case findInNode p (i + sizeNode n1 + sizeNode n2) n3 of
                Just j -> Just j
                Nothing -> findInNode p (i + sizeNode n1 + sizeNode n2 + sizeNode n3) n4

findInDigitNR :: (a -> Bool) -> Int -> Digit (Node a) -> Maybe Int
findInDigitNR p i (One n) = findInNodeR p i n
findInDigitNR p i (Two n1 n2) =
    case findInNodeR p i n2 of
        Just j -> Just j
        Nothing -> findInNodeR p (i - sizeNode n2) n1
findInDigitNR p i (Three n1 n2 n3) =
    case findInNodeR p i n3 of
        Just j -> Just j
        Nothing -> case findInNodeR p (i - sizeNode n3) n2 of
            Just j -> Just j
            Nothing -> findInNodeR p (i - sizeNode n3 - sizeNode n2) n1
findInDigitNR p i (Four n1 n2 n3 n4) =
    case findInNodeR p i n4 of
        Just j -> Just j
        Nothing -> case findInNodeR p (i - sizeNode n4) n3 of
            Just j -> Just j
            Nothing -> case findInNodeR p (i - sizeNode n4 - sizeNode n3) n2 of
                Just j -> Just j
                Nothing -> findInNodeR p (i - sizeNode n4 - sizeNode n3 - sizeNode n2) n1

-- ============================================================
-- Internal: Indexed fold helpers
-- ============================================================

foldMapDigitWithIndex :: Monoid m => (Int -> a -> m) -> Int -> Digit a -> m
foldMapDigitWithIndex f i (One a) = f i a
foldMapDigitWithIndex f i (Two a b) = f i a <> f (i+1) b
foldMapDigitWithIndex f i (Three a b c) = f i a <> f (i+1) b <> f (i+2) c
foldMapDigitWithIndex f i (Four a b c d) = f i a <> f (i+1) b <> f (i+2) c <> f (i+3) d

foldMapMiddleWithIndex :: Monoid m => (Int -> a -> m) -> Int -> Seq (Node a) -> m
foldMapMiddleWithIndex _ _ Empty = mempty
foldMapMiddleWithIndex f i (Single n) = foldMapNodeWithIndex f i n
foldMapMiddleWithIndex f i (Deep _ pr m sf) =
    foldMapDigitNWithIndex f i pr <>
    foldMapMiddleWithIndex f (i + sizeDigitN pr) m <>
    foldMapDigitNWithIndex f (i + sizeDigitN pr + sizeMiddle m) sf

foldMapNodeWithIndex :: Monoid m => (Int -> a -> m) -> Int -> Node a -> m
foldMapNodeWithIndex f i (Node2 _ a b) = f i a <> f (i+1) b
foldMapNodeWithIndex f i (Node3 _ a b c) = f i a <> f (i+1) b <> f (i+2) c

foldMapDigitNWithIndex :: Monoid m => (Int -> a -> m) -> Int -> Digit (Node a) -> m
foldMapDigitNWithIndex f i (One n) = foldMapNodeWithIndex f i n
foldMapDigitNWithIndex f i (Two n1 n2) =
    foldMapNodeWithIndex f i n1 <>
    foldMapNodeWithIndex f (i + sizeNode n1) n2
foldMapDigitNWithIndex f i (Three n1 n2 n3) =
    foldMapNodeWithIndex f i n1 <>
    foldMapNodeWithIndex f (i + sizeNode n1) n2 <>
    foldMapNodeWithIndex f (i + sizeNode n1 + sizeNode n2) n3
foldMapDigitNWithIndex f i (Four n1 n2 n3 n4) =
    foldMapNodeWithIndex f i n1 <>
    foldMapNodeWithIndex f (i + sizeNode n1) n2 <>
    foldMapNodeWithIndex f (i + sizeNode n1 + sizeNode n2) n3 <>
    foldMapNodeWithIndex f (i + sizeNode n1 + sizeNode n2 + sizeNode n3) n4

foldlDigitWithIndex :: (b -> Int -> a -> b) -> b -> Int -> Digit a -> b
foldlDigitWithIndex f z i (One a) = f z i a
foldlDigitWithIndex f z i (Two a b) = f (f z i a) (i+1) b
foldlDigitWithIndex f z i (Three a b c) = f (f (f z i a) (i+1) b) (i+2) c
foldlDigitWithIndex f z i (Four a b c d) = f (f (f (f z i a) (i+1) b) (i+2) c) (i+3) d

foldlMiddleWithIndex :: (b -> Int -> a -> b) -> b -> Int -> Seq (Node a) -> b
foldlMiddleWithIndex _ z _ Empty = z
foldlMiddleWithIndex f z i (Single n) = foldlNodeWithIndex f z i n
foldlMiddleWithIndex f z i (Deep _ pr m sf) =
    let z1 = foldlDigitNWithIndex f z i pr
        i1 = i + sizeDigitN pr
        z2 = foldlMiddleWithIndex f z1 i1 m
        i2 = i1 + sizeMiddle m
    in foldlDigitNWithIndex f z2 i2 sf

foldlNodeWithIndex :: (b -> Int -> a -> b) -> b -> Int -> Node a -> b
foldlNodeWithIndex f z i (Node2 _ a b) = f (f z i a) (i+1) b
foldlNodeWithIndex f z i (Node3 _ a b c) = f (f (f z i a) (i+1) b) (i+2) c

foldlDigitNWithIndex :: (b -> Int -> a -> b) -> b -> Int -> Digit (Node a) -> b
foldlDigitNWithIndex f z i (One n) = foldlNodeWithIndex f z i n
foldlDigitNWithIndex f z i (Two n1 n2) =
    foldlNodeWithIndex f (foldlNodeWithIndex f z i n1) (i + sizeNode n1) n2
foldlDigitNWithIndex f z i (Three n1 n2 n3) =
    let z1 = foldlNodeWithIndex f z i n1
        z2 = foldlNodeWithIndex f z1 (i + sizeNode n1) n2
    in foldlNodeWithIndex f z2 (i + sizeNode n1 + sizeNode n2) n3
foldlDigitNWithIndex f z i (Four n1 n2 n3 n4) =
    let z1 = foldlNodeWithIndex f z i n1
        z2 = foldlNodeWithIndex f z1 (i + sizeNode n1) n2
        z3 = foldlNodeWithIndex f z2 (i + sizeNode n1 + sizeNode n2) n3
    in foldlNodeWithIndex f z3 (i + sizeNode n1 + sizeNode n2 + sizeNode n3) n4

foldrDigitWithIndex :: (Int -> a -> b -> b) -> b -> Int -> Digit a -> b
foldrDigitWithIndex f z i (One a) = f i a z
foldrDigitWithIndex f z i (Two a b) = f i a (f (i+1) b z)
foldrDigitWithIndex f z i (Three a b c) = f i a (f (i+1) b (f (i+2) c z))
foldrDigitWithIndex f z i (Four a b c d) = f i a (f (i+1) b (f (i+2) c (f (i+3) d z)))

foldrMiddleWithIndex :: (Int -> a -> b -> b) -> b -> Int -> Seq (Node a) -> b
foldrMiddleWithIndex _ z _ Empty = z
foldrMiddleWithIndex f z i (Single n) = foldrNodeWithIndex f z i n
foldrMiddleWithIndex f z i (Deep _ pr m sf) =
    let i2 = i + sizeDigitN pr + sizeMiddle m
        z1 = foldrDigitNWithIndex f z i2 sf
        i1 = i + sizeDigitN pr
        z2 = foldrMiddleWithIndex f z1 i1 m
    in foldrDigitNWithIndex f z2 i pr

foldrNodeWithIndex :: (Int -> a -> b -> b) -> b -> Int -> Node a -> b
foldrNodeWithIndex f z i (Node2 _ a b) = f i a (f (i+1) b z)
foldrNodeWithIndex f z i (Node3 _ a b c) = f i a (f (i+1) b (f (i+2) c z))

foldrDigitNWithIndex :: (Int -> a -> b -> b) -> b -> Int -> Digit (Node a) -> b
foldrDigitNWithIndex f z i (One n) = foldrNodeWithIndex f z i n
foldrDigitNWithIndex f z i (Two n1 n2) =
    foldrNodeWithIndex f (foldrNodeWithIndex f z (i + sizeNode n1) n2) i n1
foldrDigitNWithIndex f z i (Three n1 n2 n3) =
    let z1 = foldrNodeWithIndex f z (i + sizeNode n1 + sizeNode n2) n3
        z2 = foldrNodeWithIndex f z1 (i + sizeNode n1) n2
    in foldrNodeWithIndex f z2 i n1
foldrDigitNWithIndex f z i (Four n1 n2 n3 n4) =
    let z1 = foldrNodeWithIndex f z (i + sizeNode n1 + sizeNode n2 + sizeNode n3) n4
        z2 = foldrNodeWithIndex f z1 (i + sizeNode n1 + sizeNode n2) n3
        z3 = foldrNodeWithIndex f z2 (i + sizeNode n1) n2
    in foldrNodeWithIndex f z3 i n1

-- ============================================================
-- Internal: Indexed traversal helpers
-- ============================================================

traverseDigitWithIndex :: Applicative f => (Int -> a -> f b) -> Int -> Digit a -> f (Digit b)
traverseDigitWithIndex f i (One a) = One <$> f i a
traverseDigitWithIndex f i (Two a b) = Two <$> f i a <*> f (i+1) b
traverseDigitWithIndex f i (Three a b c) = Three <$> f i a <*> f (i+1) b <*> f (i+2) c
traverseDigitWithIndex f i (Four a b c d) = Four <$> f i a <*> f (i+1) b <*> f (i+2) c <*> f (i+3) d

traverseMiddleWithIndex :: Applicative f => (Int -> a -> f b) -> Int -> Seq (Node a) -> f (Seq (Node b))
traverseMiddleWithIndex _ _ Empty = pure Empty
traverseMiddleWithIndex f i (Single n) = Single <$> traverseNodeWithIndex f i n
traverseMiddleWithIndex f i (Deep n pr m sf) =
    Deep n <$> traverseDigitNWithIndex f i pr
           <*> traverseMiddleWithIndex f (i + sizeDigitN pr) m
           <*> traverseDigitNWithIndex f (i + sizeDigitN pr + sizeMiddle m) sf

traverseNodeWithIndex :: Applicative f => (Int -> a -> f b) -> Int -> Node a -> f (Node b)
traverseNodeWithIndex f i (Node2 s a b) = Node2 s <$> f i a <*> f (i+1) b
traverseNodeWithIndex f i (Node3 s a b c) = Node3 s <$> f i a <*> f (i+1) b <*> f (i+2) c

traverseDigitNWithIndex :: Applicative f => (Int -> a -> f b) -> Int -> Digit (Node a) -> f (Digit (Node b))
traverseDigitNWithIndex f i (One n) = One <$> traverseNodeWithIndex f i n
traverseDigitNWithIndex f i (Two n1 n2) =
    Two <$> traverseNodeWithIndex f i n1
        <*> traverseNodeWithIndex f (i + sizeNode n1) n2
traverseDigitNWithIndex f i (Three n1 n2 n3) =
    Three <$> traverseNodeWithIndex f i n1
          <*> traverseNodeWithIndex f (i + sizeNode n1) n2
          <*> traverseNodeWithIndex f (i + sizeNode n1 + sizeNode n2) n3
traverseDigitNWithIndex f i (Four n1 n2 n3 n4) =
    Four <$> traverseNodeWithIndex f i n1
         <*> traverseNodeWithIndex f (i + sizeNode n1) n2
         <*> traverseNodeWithIndex f (i + sizeNode n1 + sizeNode n2) n3
         <*> traverseNodeWithIndex f (i + sizeNode n1 + sizeNode n2 + sizeNode n3) n4

-- ============================================================
-- Internal: Indexed map helpers
-- ============================================================

mapDigitWithIndex :: (Int -> a -> b) -> Int -> Digit a -> Digit b
mapDigitWithIndex f i (One a) = One (f i a)
mapDigitWithIndex f i (Two a b) = Two (f i a) (f (i+1) b)
mapDigitWithIndex f i (Three a b c) = Three (f i a) (f (i+1) b) (f (i+2) c)
mapDigitWithIndex f i (Four a b c d) = Four (f i a) (f (i+1) b) (f (i+2) c) (f (i+3) d)

mapMiddleWithIndex :: (Int -> a -> b) -> Int -> Seq (Node a) -> Seq (Node b)
mapMiddleWithIndex _ _ Empty = Empty
mapMiddleWithIndex f i (Single n) = Single (mapNodeWithIndex f i n)
mapMiddleWithIndex f i (Deep n pr m sf) =
    Deep n (mapDigitNWithIndex f i pr)
           (mapMiddleWithIndex f (i + sizeDigitN pr) m)
           (mapDigitNWithIndex f (i + sizeDigitN pr + sizeMiddle m) sf)

mapNodeWithIndex :: (Int -> a -> b) -> Int -> Node a -> Node b
mapNodeWithIndex f i (Node2 s a b) = Node2 s (f i a) (f (i+1) b)
mapNodeWithIndex f i (Node3 s a b c) = Node3 s (f i a) (f (i+1) b) (f (i+2) c)

mapDigitNWithIndex :: (Int -> a -> b) -> Int -> Digit (Node a) -> Digit (Node b)
mapDigitNWithIndex f i (One n) = One (mapNodeWithIndex f i n)
mapDigitNWithIndex f i (Two n1 n2) =
    Two (mapNodeWithIndex f i n1)
        (mapNodeWithIndex f (i + sizeNode n1) n2)
mapDigitNWithIndex f i (Three n1 n2 n3) =
    Three (mapNodeWithIndex f i n1)
          (mapNodeWithIndex f (i + sizeNode n1) n2)
          (mapNodeWithIndex f (i + sizeNode n1 + sizeNode n2) n3)
mapDigitNWithIndex f i (Four n1 n2 n3 n4) =
    Four (mapNodeWithIndex f i n1)
         (mapNodeWithIndex f (i + sizeNode n1) n2)
         (mapNodeWithIndex f (i + sizeNode n1 + sizeNode n2) n3)
         (mapNodeWithIndex f (i + sizeNode n1 + sizeNode n2 + sizeNode n3) n4)

-- ============================================================
-- Internal: Sorting
-- ============================================================

mergeSort :: (a -> a -> Ordering) -> [a] -> [a]
mergeSort _ [] = []
mergeSort _ [x] = [x]
mergeSort cmp xs =
    let (l, r) = Prelude.splitAt (Prelude.length xs `div` 2) xs
    in merge cmp (mergeSort cmp l) (mergeSort cmp r)

merge :: (a -> a -> Ordering) -> [a] -> [a] -> [a]
merge _ [] ys = ys
merge _ xs [] = xs
merge cmp (x:xs) (y:ys) =
    case cmp x y of
        GT -> y : merge cmp (x:xs) ys
        _  -> x : merge cmp xs (y:ys)
