-- |
-- Module      : BHC.Data.List
-- Description : List operations with guaranteed fusion
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- This module provides comprehensive list operations with guaranteed
-- fusion for common patterns. Operations marked with FUSION fuse
-- automatically when composed.
--
-- = Fusion Guarantees
--
-- The following patterns are guaranteed to fuse into a single traversal:
--
-- * @map f . map g@ fuses to @map (f . g)@
-- * @filter p . filter q@ fuses to @filter (\\x -> p x && q x)@
-- * @sum . map f@ fuses to a single accumulating loop
-- * @foldr k z . map f@ fuses to @foldr (k . f) z@

{-# LANGUAGE BangPatterns #-}

module BHC.Data.List (
    -- * Basic functions
    (++), head, last, tail, init, uncons, unsnoc,
    null, length,
    
    -- * List transformations
    -- | All transformations support fusion.
    map, reverse, intersperse, intercalate,
    transpose, subsequences, permutations,
    
    -- * Reducing lists (folds)
    -- | Strict left folds ('foldl'') are preferred for numeric accumulation.
    foldl, foldl', foldl1, foldl1',
    foldr, foldr1,
    
    -- * Special folds
    concat, concatMap,
    and, or, any, all,
    sum, product,
    maximum, minimum,
    maximumBy, minimumBy,
    
    -- * Building lists
    -- ** Scans
    scanl, scanl', scanl1,
    scanr, scanr1,
    
    -- ** Accumulating maps
    mapAccumL, mapAccumR,
    
    -- ** Infinite lists
    iterate, iterate', repeat, replicate, cycle,
    
    -- ** Unfolding
    unfoldr,
    
    -- * Sublists
    -- ** Extracting sublists
    take, drop, splitAt,
    takeWhile, dropWhile, dropWhileEnd,
    span, break, stripPrefix,
    group, groupBy,
    inits, tails,
    
    -- ** Predicates
    isPrefixOf, isSuffixOf, isInfixOf, isSubsequenceOf,
    
    -- * Searching lists
    -- ** Searching by equality
    elem, notElem, lookup,
    
    -- ** Searching with a predicate
    find, filter, partition,
    
    -- * Indexing lists
    (!?), (!!),
    elemIndex, elemIndices,
    findIndex, findIndices,
    
    -- * Zipping and unzipping lists
    zip, zip3, zip4, zip5, zip6, zip7,
    zipWith, zipWith3, zipWith4, zipWith5, zipWith6, zipWith7,
    unzip, unzip3, unzip4, unzip5, unzip6, unzip7,
    
    -- * Special lists
    -- ** Functions on strings
    lines, words, unlines, unwords,
    
    -- ** \"Set\" operations
    nub, nubBy, delete, deleteBy, (\\),
    union, unionBy, intersect, intersectBy,
    
    -- ** Ordered lists
    sort, sortBy, sortOn,
    insert, insertBy,
    
    -- * Generalized functions
    genericLength, genericTake, genericDrop,
    genericSplitAt, genericIndex, genericReplicate,
) where

import BHC.Prelude hiding (
    (++), head, last, tail, init, null, length,
    map, reverse, foldl, foldl', foldl1, foldr, foldr1,
    concat, concatMap, and, or, any, all, sum, product,
    maximum, minimum, scanl, scanr,
    take, drop, splitAt, takeWhile, dropWhile, span, break,
    elem, notElem, lookup, find, filter, partition,
    (!!), zip, zip3, zipWith, zipWith3, unzip, unzip3,
    lines, words, unlines, unwords,
    nub, delete, (\\), union, intersect, sort, sortBy, sortOn,
    iterate, repeat, replicate, cycle,
    )

-- | /O(1)/. Extract the first element of a list.
-- Throws an error on empty list.
head :: [a] -> a
head (x:_) = x
head []    = errorEmptyList "head"

-- | /O(n)/. Extract the last element of a list.
-- Throws an error on empty list.
last :: [a] -> a
last [x]    = x
last (_:xs) = last xs
last []     = errorEmptyList "last"

-- | /O(1)/. Extract the elements after the head of a list.
tail :: [a] -> [a]
tail (_:xs) = xs
tail []     = errorEmptyList "tail"

-- | /O(n)/. Return all elements except the last one.
init :: [a] -> [a]
init [_]    = []
init (x:xs) = x : init xs
init []     = errorEmptyList "init"

-- | /O(1)/. Decompose a list into its head and tail.
uncons :: [a] -> Maybe (a, [a])
uncons []     = Nothing
uncons (x:xs) = Just (x, xs)

-- | /O(n)/. Decompose a list into its init and last.
unsnoc :: [a] -> Maybe ([a], a)
unsnoc []     = Nothing
unsnoc xs     = Just (init xs, last xs)

-- | /O(1)/. Test whether a list is empty.
null :: [a] -> Bool
null []    = True
null (_:_) = False

-- | /O(n)/. Return the length of a list.
length :: [a] -> Int
length = foldl' (\c _ -> c + 1) 0

-- | /O(n)/. Append two lists.
-- FUSION: Producer
(++) :: [a] -> [a] -> [a]
[]     ++ ys = ys
(x:xs) ++ ys = x : (xs ++ ys)
infixr 5 ++

-- | /O(n)/. Apply a function to each element.
-- FUSION: map/map, map/filter, fold/map all fuse.
map :: (a -> b) -> [a] -> [b]
map _ []     = []
map f (x:xs) = f x : map f xs

-- | /O(n)/. Reverse a list.
reverse :: [a] -> [a]
reverse = foldl' (flip (:)) []

-- | /O(n)/. Insert an element between each pair.
intersperse :: a -> [a] -> [a]
intersperse _   []     = []
intersperse _   [x]    = [x]
intersperse sep (x:xs) = x : sep : intersperse sep xs

-- | /O(n)/. Insert a list between each pair of lists.
intercalate :: [a] -> [[a]] -> [a]
intercalate xs xss = concat (intersperse xs xss)

-- | Transpose rows and columns.
transpose :: [[a]] -> [[a]]
transpose []             = []
transpose ([]     : xss) = transpose xss
transpose ((x:xs) : xss) = (x : [h | (h:_) <- xss]) 
                         : transpose (xs : [t | (_:t) <- xss])

-- | All subsequences (power set).
subsequences :: [a] -> [[a]]
subsequences xs = [] : nonEmptySubsequences xs
  where
    nonEmptySubsequences []     = []
    nonEmptySubsequences (y:ys) = [y] : foldr f [] (nonEmptySubsequences ys)
      where f zs r = zs : (y : zs) : r

-- | All permutations.
permutations :: [a] -> [[a]]
permutations xs0 = xs0 : perms xs0 []
  where
    perms []     _  = []
    perms (t:ts) is = foldr interleave (perms ts (t:is)) (permutations is)
      where
        interleave xs r = let (_,zs) = interleave' id xs r in zs
        interleave' _ []     r = (ts, r)
        interleave' f (y:ys) r = 
            let (us,zs) = interleave' (f . (y:)) ys r
            in  (y:us, f (t:y:us) : zs)

-- Folds
foldl :: (b -> a -> b) -> b -> [a] -> b
foldl _ z []     = z
foldl f z (x:xs) = foldl f (f z x) xs

foldl' :: (b -> a -> b) -> b -> [a] -> b
foldl' _ z []     = z
foldl' f z (x:xs) = let z' = f z x in z' `seq` foldl' f z' xs

foldl1 :: (a -> a -> a) -> [a] -> a
foldl1 f (x:xs) = foldl f x xs
foldl1 _ []     = errorEmptyList "foldl1"

foldl1' :: (a -> a -> a) -> [a] -> a
foldl1' f (x:xs) = foldl' f x xs
foldl1' _ []     = errorEmptyList "foldl1'"

foldr :: (a -> b -> b) -> b -> [a] -> b
foldr _ z []     = z
foldr f z (x:xs) = f x (foldr f z xs)

foldr1 :: (a -> a -> a) -> [a] -> a
foldr1 _ [x]    = x
foldr1 f (x:xs) = f x (foldr1 f xs)
foldr1 _ []     = errorEmptyList "foldr1"

-- Special folds
concat :: [[a]] -> [a]
concat = foldr (++) []

concatMap :: (a -> [b]) -> [a] -> [b]
concatMap f = foldr ((++) . f) []

and :: [Bool] -> Bool
and = foldr (&&) True

or :: [Bool] -> Bool
or = foldr (||) False

any :: (a -> Bool) -> [a] -> Bool
any p = or . map p

all :: (a -> Bool) -> [a] -> Bool
all p = and . map p

sum :: Num a => [a] -> a
sum = foldl' (+) 0

product :: Num a => [a] -> a
product = foldl' (*) 1

maximum :: Ord a => [a] -> a
maximum = foldl1 max

minimum :: Ord a => [a] -> a
minimum = foldl1 min

maximumBy :: (a -> a -> Ordering) -> [a] -> a
maximumBy cmp = foldl1 (\x y -> if cmp x y == GT then x else y)

minimumBy :: (a -> a -> Ordering) -> [a] -> a
minimumBy cmp = foldl1 (\x y -> if cmp x y == LT then x else y)

-- Scans
scanl :: (b -> a -> b) -> b -> [a] -> [b]
scanl f q ls = q : case ls of
    []   -> []
    x:xs -> scanl f (f q x) xs

scanl' :: (b -> a -> b) -> b -> [a] -> [b]
scanl' f q ls = q : case ls of
    []   -> []
    x:xs -> let q' = f q x in q' `seq` scanl' f q' xs

scanl1 :: (a -> a -> a) -> [a] -> [a]
scanl1 _ []     = []
scanl1 f (x:xs) = scanl f x xs

scanr :: (a -> b -> b) -> b -> [a] -> [b]
scanr _ q0 []     = [q0]
scanr f q0 (x:xs) = f x q : qs
  where qs@(q:_) = scanr f q0 xs

scanr1 :: (a -> a -> a) -> [a] -> [a]
scanr1 _ []     = []
scanr1 _ [x]    = [x]
scanr1 f (x:xs) = f x q : qs
  where qs@(q:_) = scanr1 f xs

-- Accumulating maps
mapAccumL :: (acc -> x -> (acc, y)) -> acc -> [x] -> (acc, [y])
mapAccumL _ s []     = (s, [])
mapAccumL f s (x:xs) = (s'', y:ys)
  where (s',  y)  = f s x
        (s'', ys) = mapAccumL f s' xs

mapAccumR :: (acc -> x -> (acc, y)) -> acc -> [x] -> (acc, [y])
mapAccumR _ s []     = (s, [])
mapAccumR f s (x:xs) = (s'', y:ys)
  where (s'', y)  = f s' x
        (s',  ys) = mapAccumR f s xs

-- Infinite lists
iterate :: (a -> a) -> a -> [a]
iterate f x = x : iterate f (f x)

iterate' :: (a -> a) -> a -> [a]
iterate' f x = x `seq` (x : iterate' f (f x))

repeat :: a -> [a]
repeat x = xs where xs = x : xs

replicate :: Int -> a -> [a]
replicate n x = take n (repeat x)

cycle :: [a] -> [a]
cycle [] = errorEmptyList "cycle"
cycle xs = xs' where xs' = xs ++ xs'

unfoldr :: (b -> Maybe (a, b)) -> b -> [a]
unfoldr f b = case f b of
    Just (a, b') -> a : unfoldr f b'
    Nothing      -> []

-- Sublists
take :: Int -> [a] -> [a]
take n _ | n <= 0 = []
take _ []         = []
take n (x:xs)     = x : take (n - 1) xs

drop :: Int -> [a] -> [a]
drop n xs | n <= 0 = xs
drop _ []          = []
drop n (_:xs)      = drop (n - 1) xs

splitAt :: Int -> [a] -> ([a], [a])
splitAt n xs = (take n xs, drop n xs)

takeWhile :: (a -> Bool) -> [a] -> [a]
takeWhile _ []     = []
takeWhile p (x:xs)
    | p x       = x : takeWhile p xs
    | otherwise = []

dropWhile :: (a -> Bool) -> [a] -> [a]
dropWhile _ []     = []
dropWhile p xs@(x:xs')
    | p x       = dropWhile p xs'
    | otherwise = xs

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = foldr (\x xs -> if p x && null xs then [] else x:xs) []

span :: (a -> Bool) -> [a] -> ([a], [a])
span _ []     = ([], [])
span p xs@(x:xs')
    | p x       = let (ys, zs) = span p xs' in (x:ys, zs)
    | otherwise = ([], xs)

break :: (a -> Bool) -> [a] -> ([a], [a])
break p = span (not . p)

stripPrefix :: Eq a => [a] -> [a] -> Maybe [a]
stripPrefix [] ys         = Just ys
stripPrefix (x:xs) (y:ys)
    | x == y              = stripPrefix xs ys
stripPrefix _ _           = Nothing

group :: Eq a => [a] -> [[a]]
group = groupBy (==)

groupBy :: (a -> a -> Bool) -> [a] -> [[a]]
groupBy _  []     = []
groupBy eq (x:xs) = (x:ys) : groupBy eq zs
  where (ys, zs) = span (eq x) xs

inits :: [a] -> [[a]]
inits []     = [[]]
inits (x:xs) = [] : map (x:) (inits xs)

tails :: [a] -> [[a]]
tails []         = [[]]
tails xs@(_:xs') = xs : tails xs'

isPrefixOf :: Eq a => [a] -> [a] -> Bool
isPrefixOf [] _          = True
isPrefixOf _  []         = False
isPrefixOf (x:xs) (y:ys) = x == y && isPrefixOf xs ys

isSuffixOf :: Eq a => [a] -> [a] -> Bool
isSuffixOf x y = isPrefixOf (reverse x) (reverse y)

isInfixOf :: Eq a => [a] -> [a] -> Bool
isInfixOf needle haystack = any (isPrefixOf needle) (tails haystack)

isSubsequenceOf :: Eq a => [a] -> [a] -> Bool
isSubsequenceOf []     _      = True
isSubsequenceOf _      []     = False
isSubsequenceOf a@(x:xs) (y:ys)
    | x == y    = isSubsequenceOf xs ys
    | otherwise = isSubsequenceOf a ys

-- Searching
elem :: Eq a => a -> [a] -> Bool
elem _ []     = False
elem x (y:ys) = x == y || elem x ys

notElem :: Eq a => a -> [a] -> Bool
notElem x = not . elem x

lookup :: Eq a => a -> [(a, b)] -> Maybe b
lookup _ []          = Nothing
lookup k ((x, v):xs)
    | k == x         = Just v
    | otherwise      = lookup k xs

find :: (a -> Bool) -> [a] -> Maybe a
find _ []     = Nothing
find p (x:xs)
    | p x       = Just x
    | otherwise = find p xs

filter :: (a -> Bool) -> [a] -> [a]
filter _ []     = []
filter p (x:xs)
    | p x       = x : filter p xs
    | otherwise = filter p xs

partition :: (a -> Bool) -> [a] -> ([a], [a])
partition p = foldr select ([], [])
  where select x (ts, fs) | p x       = (x:ts, fs)
                          | otherwise = (ts, x:fs)

-- Indexing
(!?) :: [a] -> Int -> Maybe a
[]     !? _ = Nothing
(x:_)  !? 0 = Just x
(_:xs) !? n | n > 0     = xs !? (n - 1)
            | otherwise = Nothing
infixl 9 !?

(!!) :: [a] -> Int -> a
xs !! n | n < 0 = error "!!: negative index"
[]     !! _     = error "!!: index too large"
(x:_)  !! 0     = x
(_:xs) !! n     = xs !! (n - 1)
infixl 9 !!

elemIndex :: Eq a => a -> [a] -> Maybe Int
elemIndex x = findIndex (== x)

elemIndices :: Eq a => a -> [a] -> [Int]
elemIndices x = findIndices (== x)

findIndex :: (a -> Bool) -> [a] -> Maybe Int
findIndex p = go 0
  where go _ []     = Nothing
        go i (x:xs) | p x       = Just i
                    | otherwise = go (i + 1) xs

findIndices :: (a -> Bool) -> [a] -> [Int]
findIndices p xs = [i | (x, i) <- zip xs [0..], p x]

-- Zipping
zip :: [a] -> [b] -> [(a, b)]
zip = zipWith (,)

zip3 :: [a] -> [b] -> [c] -> [(a, b, c)]
zip3 = zipWith3 (,,)

zip4 :: [a] -> [b] -> [c] -> [d] -> [(a, b, c, d)]
zip4 = zipWith4 (,,,)

zip5 :: [a] -> [b] -> [c] -> [d] -> [e] -> [(a, b, c, d, e)]
zip5 = zipWith5 (,,,,)

zip6 :: [a] -> [b] -> [c] -> [d] -> [e] -> [f] -> [(a, b, c, d, e, f)]
zip6 = zipWith6 (,,,,,)

zip7 :: [a] -> [b] -> [c] -> [d] -> [e] -> [f] -> [g] -> [(a, b, c, d, e, f, g)]
zip7 = zipWith7 (,,,,,,)

zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith _ []     _      = []
zipWith _ _      []     = []
zipWith f (x:xs) (y:ys) = f x y : zipWith f xs ys

zipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
zipWith3 _ []     _      _      = []
zipWith3 _ _      []     _      = []
zipWith3 _ _      _      []     = []
zipWith3 f (x:xs) (y:ys) (z:zs) = f x y z : zipWith3 f xs ys zs

zipWith4 :: (a -> b -> c -> d -> e) -> [a] -> [b] -> [c] -> [d] -> [e]
zipWith4 f as bs cs ds = zipWith (\a (b,c,d) -> f a b c d) as (zip3 bs cs ds)

zipWith5 :: (a -> b -> c -> d -> e -> f) -> [a] -> [b] -> [c] -> [d] -> [e] -> [f]
zipWith5 f as bs cs ds es = zipWith (\a (b,c,d,e) -> f a b c d e) as (zip4 bs cs ds es)

zipWith6 :: (a -> b -> c -> d -> e -> f -> g) -> [a] -> [b] -> [c] -> [d] -> [e] -> [f] -> [g]
zipWith6 fn as bs cs ds es fs = zipWith (\a (b,c,d,e,f) -> fn a b c d e f) as (zip5 bs cs ds es fs)

zipWith7 :: (a -> b -> c -> d -> e -> f -> g -> h) -> [a] -> [b] -> [c] -> [d] -> [e] -> [f] -> [g] -> [h]
zipWith7 fn as bs cs ds es fs gs = zipWith (\a (b,c,d,e,f,g) -> fn a b c d e f g) as (zip6 bs cs ds es fs gs)

unzip :: [(a, b)] -> ([a], [b])
unzip = foldr (\(a, b) (as, bs) -> (a:as, b:bs)) ([], [])

unzip3 :: [(a, b, c)] -> ([a], [b], [c])
unzip3 = foldr (\(a, b, c) (as, bs, cs) -> (a:as, b:bs, c:cs)) ([], [], [])

unzip4 :: [(a, b, c, d)] -> ([a], [b], [c], [d])
unzip4 = foldr (\(a, b, c, d) (as, bs, cs, ds) -> (a:as, b:bs, c:cs, d:ds)) ([], [], [], [])

unzip5 :: [(a, b, c, d, e)] -> ([a], [b], [c], [d], [e])
unzip5 = foldr (\(a, b, c, d, e) (as, bs, cs, ds, es) -> (a:as, b:bs, c:cs, d:ds, e:es)) ([], [], [], [], [])

unzip6 :: [(a, b, c, d, e, f)] -> ([a], [b], [c], [d], [e], [f])
unzip6 = foldr (\(a, b, c, d, e, f) (as, bs, cs, ds, es, fs) -> (a:as, b:bs, c:cs, d:ds, e:es, f:fs)) ([], [], [], [], [], [])

unzip7 :: [(a, b, c, d, e, f, g)] -> ([a], [b], [c], [d], [e], [f], [g])
unzip7 = foldr (\(a, b, c, d, e, f, g) (as, bs, cs, ds, es, fs, gs) -> (a:as, b:bs, c:cs, d:ds, e:es, f:fs, g:gs)) ([], [], [], [], [], [], [])

-- String functions
lines :: String -> [String]
lines ""   = []
lines s    = let (l, s') = break (== '\n') s
             in l : case s' of
                      []      -> []
                      (_:s'') -> lines s''

words :: String -> [String]
words s = case dropWhile isSpace s of
    "" -> []
    s' -> w : words s''
      where (w, s'') = break isSpace s'
  where isSpace c = c `elem` " \t\n\r"

unlines :: [String] -> String
unlines = concatMap (++ "\n")

unwords :: [String] -> String
unwords []     = ""
unwords [w]    = w
unwords (w:ws) = w ++ ' ' : unwords ws

-- Set operations
nub :: Eq a => [a] -> [a]
nub = nubBy (==)

nubBy :: (a -> a -> Bool) -> [a] -> [a]
nubBy eq = go []
  where go _ []     = []
        go seen (x:xs)
            | elemBy eq x seen = go seen xs
            | otherwise        = x : go (x:seen) xs
        elemBy f y = any (f y)

delete :: Eq a => a -> [a] -> [a]
delete = deleteBy (==)

deleteBy :: (a -> a -> Bool) -> a -> [a] -> [a]
deleteBy _  _ []     = []
deleteBy eq x (y:ys)
    | x `eq` y       = ys
    | otherwise      = y : deleteBy eq x ys

(\\) :: Eq a => [a] -> [a] -> [a]
(\\) = foldl (flip delete)
infixl 9 \\

union :: Eq a => [a] -> [a] -> [a]
union = unionBy (==)

unionBy :: (a -> a -> Bool) -> [a] -> [a] -> [a]
unionBy eq xs ys = xs ++ foldl (flip (deleteBy eq)) (nubBy eq ys) xs

intersect :: Eq a => [a] -> [a] -> [a]
intersect = intersectBy (==)

intersectBy :: (a -> a -> Bool) -> [a] -> [a] -> [a]
intersectBy eq xs ys = [x | x <- xs, elemBy eq x ys]
  where elemBy f y = any (f y)

-- Sorting
sort :: Ord a => [a] -> [a]
sort = sortBy compare

sortBy :: (a -> a -> Ordering) -> [a] -> [a]
sortBy cmp = mergeAll . sequences
  where
    sequences (a:b:xs)
      | cmp a b == GT = descending b [a] xs
      | otherwise     = ascending b (a:) xs
    sequences xs      = [xs]
    
    descending a as (b:bs)
      | cmp a b == GT = descending b (a:as) bs
    descending a as bs    = (a:as) : sequences bs
    
    ascending a as (b:bs)
      | cmp a b /= GT = ascending b (\ys -> as (a:ys)) bs
    ascending a as bs     = let !x = as [a] in x : sequences bs
    
    mergeAll [x] = x
    mergeAll []  = []
    mergeAll xs  = mergeAll (mergePairs xs)
    
    mergePairs (a:b:xs) = let !x = merge a b in x : mergePairs xs
    mergePairs xs       = xs
    
    merge as@(a:as') bs@(b:bs')
      | cmp a b == GT = b : merge as bs'
      | otherwise     = a : merge as' bs
    merge [] bs       = bs
    merge as []       = as

sortOn :: Ord b => (a -> b) -> [a] -> [a]
sortOn f = map snd . sortBy (compare `on` fst) . map (\x -> (f x, x))

insert :: Ord a => a -> [a] -> [a]
insert = insertBy compare

insertBy :: (a -> a -> Ordering) -> a -> [a] -> [a]
insertBy _   x [] = [x]
insertBy cmp x ys@(y:ys')
    | cmp x y == GT = y : insertBy cmp x ys'
    | otherwise     = x : ys

-- Generic functions
genericLength :: Num i => [a] -> i
genericLength []     = 0
genericLength (_:xs) = 1 + genericLength xs

genericTake :: Integral i => i -> [a] -> [a]
genericTake n _ | n <= 0 = []
genericTake _ []         = []
genericTake n (x:xs)     = x : genericTake (n - 1) xs

genericDrop :: Integral i => i -> [a] -> [a]
genericDrop n xs | n <= 0 = xs
genericDrop _ []          = []
genericDrop n (_:xs)      = genericDrop (n - 1) xs

genericSplitAt :: Integral i => i -> [a] -> ([a], [a])
genericSplitAt n xs = (genericTake n xs, genericDrop n xs)

genericIndex :: Integral i => [a] -> i -> a
genericIndex (x:_)  0 = x
genericIndex (_:xs) n | n > 0     = genericIndex xs (n - 1)
                      | otherwise = error "genericIndex: negative index"
genericIndex []     _ = error "genericIndex: index too large"

genericReplicate :: Integral i => i -> a -> [a]
genericReplicate n x = genericTake n (repeat x)

-- Error helper
errorEmptyList :: String -> a
errorEmptyList fun = error ("BHC.Data.List." ++ fun ++ ": empty list")
