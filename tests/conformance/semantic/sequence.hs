-- Test: sequence
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Phase 3 - Containers

{-# HASKELL_EDITION 2026 #-}

module SequenceTest where

import qualified BHC.Data.Sequence as Seq
import BHC.Data.Sequence (Seq, ViewL(..), ViewR(..), (<|), (|>), (><))

-- ================================================================
-- Construction Tests
-- ================================================================

testEmpty :: Bool
testEmpty = Seq.null Seq.empty && Seq.length Seq.empty == 0
-- Result: True

testSingleton :: Bool
testSingleton =
    let s = Seq.singleton 42
    in Seq.length s == 1 && Seq.index s 0 == 42
-- Result: True

testFromList :: Bool
testFromList =
    let s = Seq.fromList [1, 2, 3, 4, 5]
    in Seq.length s == 5
-- Result: True

testFromFunction :: Bool
testFromFunction =
    let s = Seq.fromFunction 5 (* 2)  -- [0, 2, 4, 6, 8]
    in Seq.length s == 5 && Seq.index s 2 == 4
-- Result: True

testReplicate :: Bool
testReplicate =
    let s = Seq.replicate 3 "x"
    in Seq.length s == 3 && Seq.index s 1 == "x"
-- Result: True

-- ================================================================
-- Cons/Snoc Tests
-- ================================================================

testConsLeft :: Bool
testConsLeft =
    let s1 = Seq.fromList [2, 3, 4]
        s2 = 1 <| s1
    in Seq.length s2 == 4 && Seq.index s2 0 == 1
-- Result: True

testConsRight :: Bool
testConsRight =
    let s1 = Seq.fromList [1, 2, 3]
        s2 = s1 |> 4
    in Seq.length s2 == 4 && Seq.index s2 3 == 4
-- Result: True

testConcat :: Bool
testConcat =
    let s1 = Seq.fromList [1, 2, 3]
        s2 = Seq.fromList [4, 5, 6]
        s3 = s1 >< s2
    in Seq.length s3 == 6 && Seq.index s3 3 == 4
-- Result: True

-- ================================================================
-- Deconstruction Tests
-- ================================================================

testNull :: Bool
testNull =
    Seq.null Seq.empty && not (Seq.null (Seq.singleton 1))
-- Result: True

testLength :: Bool
testLength =
    Seq.length Seq.empty == 0 &&
    Seq.length (Seq.singleton 1) == 1 &&
    Seq.length (Seq.fromList [1, 2, 3]) == 3
-- Result: True

testViewL :: Bool
testViewL =
    case Seq.viewl (Seq.fromList [1, 2, 3]) of
        x :< xs -> x == 1 && Seq.length xs == 2
        EmptyL -> False
-- Result: True

testViewLEmpty :: Bool
testViewLEmpty =
    case Seq.viewl Seq.empty of
        EmptyL -> True
        _ -> False
-- Result: True

testViewR :: Bool
testViewR =
    case Seq.viewr (Seq.fromList [1, 2, 3]) of
        xs :> x -> x == 3 && Seq.length xs == 2
        EmptyR -> False
-- Result: True

testViewREmpty :: Bool
testViewREmpty =
    case Seq.viewr Seq.empty of
        EmptyR -> True
        _ -> False
-- Result: True

-- ================================================================
-- Indexing Tests
-- ================================================================

testIndex :: Bool
testIndex =
    let s = Seq.fromList [10, 20, 30, 40, 50]
    in Seq.index s 0 == 10 &&
       Seq.index s 2 == 30 &&
       Seq.index s 4 == 50
-- Result: True

testLookup :: Bool
testLookup =
    let s = Seq.fromList [1, 2, 3]
    in Seq.lookup 1 s == Just 2 &&
       Seq.lookup 5 s == Nothing &&
       Seq.lookup (-1) s == Nothing
-- Result: True

testInfixLookup :: Bool
testInfixLookup =
    let s = Seq.fromList ['a', 'b', 'c']
    in (s Seq.!? 1) == Just 'b' && (s Seq.!? 10) == Nothing
-- Result: True

testAdjust :: Bool
testAdjust =
    let s1 = Seq.fromList [1, 2, 3]
        s2 = Seq.adjust (* 10) 1 s1
    in Seq.index s2 1 == 20
-- Result: True

testUpdate :: Bool
testUpdate =
    let s1 = Seq.fromList ["a", "b", "c"]
        s2 = Seq.update 1 "B" s1
    in Seq.index s2 1 == "B"
-- Result: True

testTake :: Bool
testTake =
    let s = Seq.fromList [1, 2, 3, 4, 5]
    in Seq.length (Seq.take 3 s) == 3 &&
       Seq.length (Seq.take 0 s) == 0 &&
       Seq.length (Seq.take 10 s) == 5
-- Result: True

testDrop :: Bool
testDrop =
    let s = Seq.fromList [1, 2, 3, 4, 5]
    in Seq.length (Seq.drop 2 s) == 3 &&
       Seq.index (Seq.drop 2 s) 0 == 3
-- Result: True

testSplitAt :: Bool
testSplitAt =
    let s = Seq.fromList [1, 2, 3, 4, 5]
        (l, r) = Seq.splitAt 2 s
    in Seq.length l == 2 && Seq.length r == 3
-- Result: True

testInsertAt :: Bool
testInsertAt =
    let s1 = Seq.fromList [1, 3, 4]
        s2 = Seq.insertAt 1 2 s1
    in Seq.length s2 == 4 && Seq.index s2 1 == 2
-- Result: True

testDeleteAt :: Bool
testDeleteAt =
    let s1 = Seq.fromList [1, 2, 3, 4]
        s2 = Seq.deleteAt 1 s1
    in Seq.length s2 == 3 && Seq.index s2 1 == 3
-- Result: True

-- ================================================================
-- Search Tests
-- ================================================================

testElemIndexL :: Bool
testElemIndexL =
    let s = Seq.fromList [1, 2, 3, 2, 1]
    in Seq.elemIndexL 2 s == Just 1  -- First occurrence
-- Result: True

testElemIndexR :: Bool
testElemIndexR =
    let s = Seq.fromList [1, 2, 3, 2, 1]
    in Seq.elemIndexR 2 s == Just 3  -- Last occurrence
-- Result: True

testElemIndicesL :: Bool
testElemIndicesL =
    let s = Seq.fromList [1, 2, 1, 2, 1]
    in Seq.elemIndicesL 1 s == [0, 2, 4]
-- Result: True

testFindIndexL :: Bool
testFindIndexL =
    let s = Seq.fromList [1, 3, 5, 7, 9]
    in Seq.findIndexL (> 4) s == Just 2  -- First element > 4
-- Result: True

testFindIndexR :: Bool
testFindIndexR =
    let s = Seq.fromList [1, 3, 5, 7, 9]
    in Seq.findIndexR (< 6) s == Just 2  -- Last element < 6
-- Result: True

-- ================================================================
-- Transformation Tests
-- ================================================================

testReverse :: Bool
testReverse =
    let s1 = Seq.fromList [1, 2, 3, 4, 5]
        s2 = Seq.reverse s1
    in Seq.index s2 0 == 5 && Seq.index s2 4 == 1
-- Result: True

testIntersperse :: Bool
testIntersperse =
    let s1 = Seq.fromList [1, 2, 3]
        s2 = Seq.intersperse 0 s1
    in Seq.length s2 == 5  -- [1, 0, 2, 0, 3]
-- Result: True

testScanl :: Bool
testScanl =
    let s1 = Seq.fromList [1, 2, 3]
        s2 = Seq.scanl (+) 0 s1
    in Seq.length s2 == 4  -- [0, 1, 3, 6]
-- Result: True

testScanr :: Bool
testScanr =
    let s1 = Seq.fromList [1, 2, 3]
        s2 = Seq.scanr (+) 0 s1
    in Seq.length s2 == 4  -- [6, 5, 3, 0]
-- Result: True

-- ================================================================
-- Sorting Tests
-- ================================================================

testSort :: Bool
testSort =
    let s1 = Seq.fromList [5, 2, 8, 1, 9, 3]
        s2 = Seq.sort s1
    in Seq.index s2 0 == 1 && Seq.index s2 5 == 9
-- Result: True

testSortBy :: Bool
testSortBy =
    let s1 = Seq.fromList [1, 2, 3, 4, 5]
        s2 = Seq.sortBy (flip compare) s1  -- Descending
    in Seq.index s2 0 == 5 && Seq.index s2 4 == 1
-- Result: True

testSortOn :: Bool
testSortOn =
    let s1 = Seq.fromList ["bb", "a", "ccc"]
        s2 = Seq.sortOn length s1
    in Seq.index s2 0 == "a" && Seq.index s2 2 == "ccc"
-- Result: True

-- ================================================================
-- Subsequence Tests
-- ================================================================

testTails :: Bool
testTails =
    let s = Seq.fromList [1, 2, 3]
        ts = Seq.tails s
    in Seq.length ts == 4  -- Original plus 3 suffixes plus empty
-- Result: True

testInits :: Bool
testInits =
    let s = Seq.fromList [1, 2, 3]
        is = Seq.inits s
    in Seq.length is == 4  -- Empty plus 3 prefixes
-- Result: True

testChunksOf :: Bool
testChunksOf =
    let s = Seq.fromList [1, 2, 3, 4, 5, 6, 7]
        cs = Seq.chunksOf 3 s
    in Seq.length cs == 3  -- [1,2,3], [4,5,6], [7]
-- Result: True

-- ================================================================
-- Filter Tests
-- ================================================================

testTakeWhileL :: Bool
testTakeWhileL =
    let s1 = Seq.fromList [1, 2, 3, 4, 5]
        s2 = Seq.takeWhileL (< 4) s1
    in Seq.length s2 == 3  -- [1, 2, 3]
-- Result: True

testTakeWhileR :: Bool
testTakeWhileR =
    let s1 = Seq.fromList [1, 2, 3, 4, 5]
        s2 = Seq.takeWhileR (> 2) s1
    in Seq.length s2 == 3  -- [3, 4, 5]
-- Result: True

testDropWhileL :: Bool
testDropWhileL =
    let s1 = Seq.fromList [1, 2, 3, 4, 5]
        s2 = Seq.dropWhileL (< 3) s1
    in Seq.length s2 == 3 && Seq.index s2 0 == 3
-- Result: True

testDropWhileR :: Bool
testDropWhileR =
    let s1 = Seq.fromList [1, 2, 3, 4, 5]
        s2 = Seq.dropWhileR (> 3) s1
    in Seq.length s2 == 3 && Seq.index s2 2 == 3
-- Result: True

testSpanl :: Bool
testSpanl =
    let s = Seq.fromList [1, 2, 3, 4, 5]
        (yes, no) = Seq.spanl (< 4) s
    in Seq.length yes == 3 && Seq.length no == 2
-- Result: True

testBreakl :: Bool
testBreakl =
    let s = Seq.fromList [1, 2, 3, 4, 5]
        (before, after) = Seq.breakl (>= 4) s
    in Seq.length before == 3 && Seq.length after == 2
-- Result: True

testPartition :: Bool
testPartition =
    let s = Seq.fromList [1, 2, 3, 4, 5, 6]
        (evens, odds) = Seq.partition even s
    in Seq.length evens == 3 && Seq.length odds == 3
-- Result: True

testFilter :: Bool
testFilter =
    let s1 = Seq.fromList [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        s2 = Seq.filter even s1
    in Seq.length s2 == 5
-- Result: True

-- ================================================================
-- Zip Tests
-- ================================================================

testZip :: Bool
testZip =
    let s1 = Seq.fromList [1, 2, 3]
        s2 = Seq.fromList ['a', 'b', 'c']
        s3 = Seq.zip s1 s2
    in Seq.length s3 == 3 && Seq.index s3 0 == (1, 'a')
-- Result: True

testZipWith :: Bool
testZipWith =
    let s1 = Seq.fromList [1, 2, 3]
        s2 = Seq.fromList [10, 20, 30]
        s3 = Seq.zipWith (+) s1 s2
    in Seq.index s3 0 == 11 && Seq.index s3 2 == 33
-- Result: True

testZipDifferentLengths :: Bool
testZipDifferentLengths =
    let s1 = Seq.fromList [1, 2, 3, 4, 5]
        s2 = Seq.fromList ['a', 'b', 'c']
        s3 = Seq.zip s1 s2
    in Seq.length s3 == 3  -- Truncated to shorter
-- Result: True

testUnzip :: Bool
testUnzip =
    let s = Seq.fromList [(1, 'a'), (2, 'b'), (3, 'c')]
        (s1, s2) = Seq.unzip s
    in Seq.length s1 == 3 && Seq.index s1 1 == 2 && Seq.index s2 1 == 'b'
-- Result: True

-- ================================================================
-- Fold Tests
-- ================================================================

testFoldMapWithIndex :: Bool
testFoldMapWithIndex =
    let s = Seq.fromList [10, 20, 30]
        result = Seq.foldMapWithIndex (\i x -> [i + x]) s
    in result == [10, 21, 32]
-- Result: True

testFoldlWithIndex :: Bool
testFoldlWithIndex =
    let s = Seq.fromList [1, 2, 3]
        result = Seq.foldlWithIndex (\acc i x -> acc + i * x) 0 s
    in result == 0*1 + 1*2 + 2*3  -- 0 + 2 + 6 = 8
-- Result: True

testFoldrWithIndex :: Bool
testFoldrWithIndex =
    let s = Seq.fromList [1, 2, 3]
        result = Seq.foldrWithIndex (\i x acc -> i + x + acc) 0 s
    in result == (0+1) + (1+2) + (2+3)  -- 1 + 3 + 5 = 9
-- Result: True

-- ================================================================
-- Map Tests
-- ================================================================

testMapWithIndex :: Bool
testMapWithIndex =
    let s1 = Seq.fromList [10, 20, 30]
        s2 = Seq.mapWithIndex (\i x -> i + x) s1
    in Seq.index s2 0 == 10 && Seq.index s2 1 == 21 && Seq.index s2 2 == 32
-- Result: True

-- ================================================================
-- Edge Cases
-- ================================================================

testEmptyOperations :: Bool
testEmptyOperations =
    let e = Seq.empty
    in Seq.null e &&
       Seq.length e == 0 &&
       Seq.lookup 0 e == Nothing &&
       Seq.take 5 e == e &&
       Seq.drop 5 e == e
-- Result: True

testSingleElement :: Bool
testSingleElement =
    let s = Seq.singleton 42
    in Seq.length s == 1 &&
       Seq.index s 0 == 42 &&
       case Seq.viewl s of { x :< xs -> x == 42 && Seq.null xs; _ -> False } &&
       case Seq.viewr s of { xs :> x -> x == 42 && Seq.null xs; _ -> False }
-- Result: True

testLargeSequence :: Bool
testLargeSequence =
    let s = Seq.fromList [1..1000]
    in Seq.length s == 1000 &&
       Seq.index s 0 == 1 &&
       Seq.index s 999 == 1000 &&
       Seq.index s 500 == 501
-- Result: True

-- ================================================================
-- Property-style Tests
-- ================================================================

-- fromList then toList is identity (via Foldable)
propFromToList :: Bool
propFromToList =
    let xs = [1, 2, 3, 4, 5]
        s = Seq.fromList xs
        ys = foldr (:) [] s  -- toList via Foldable
    in xs == ys
-- Result: True

-- Concatenation is associative
propConcatAssoc :: Bool
propConcatAssoc =
    let a = Seq.fromList [1, 2]
        b = Seq.fromList [3, 4]
        c = Seq.fromList [5, 6]
    in ((a >< b) >< c) == (a >< (b >< c))
-- Result: True (comparing via toList)

-- Empty is identity for concat
propConcatEmpty :: Bool
propConcatEmpty =
    let s = Seq.fromList [1, 2, 3]
    in (Seq.empty >< s) == s && (s >< Seq.empty) == s
-- Result: True

-- Reverse twice is identity
propReverseReverse :: Bool
propReverseReverse =
    let s = Seq.fromList [1, 2, 3, 4, 5]
    in Seq.reverse (Seq.reverse s) == s
-- Result: True

-- Take and drop partition the sequence
propTakeDropSplit :: Bool
propTakeDropSplit =
    let s = Seq.fromList [1, 2, 3, 4, 5]
        n = 3
    in (Seq.take n s >< Seq.drop n s) == s
-- Result: True

-- Length of cons is +1
propConsLength :: Bool
propConsLength =
    let s = Seq.fromList [1, 2, 3]
    in Seq.length (0 <| s) == Seq.length s + 1 &&
       Seq.length (s |> 4) == Seq.length s + 1
-- Result: True

-- ================================================================
-- Main
-- ================================================================

main :: IO ()
main = do
    -- Construction
    print testEmpty
    print testSingleton
    print testFromList
    print testFromFunction
    print testReplicate

    -- Cons/Snoc
    print testConsLeft
    print testConsRight
    print testConcat

    -- Deconstruction
    print testNull
    print testLength
    print testViewL
    print testViewLEmpty
    print testViewR
    print testViewREmpty

    -- Indexing
    print testIndex
    print testLookup
    print testInfixLookup
    print testAdjust
    print testUpdate
    print testTake
    print testDrop
    print testSplitAt
    print testInsertAt
    print testDeleteAt

    -- Search
    print testElemIndexL
    print testElemIndexR
    print testElemIndicesL
    print testFindIndexL
    print testFindIndexR

    -- Transformation
    print testReverse
    print testIntersperse
    print testScanl
    print testScanr

    -- Sorting
    print testSort
    print testSortBy
    print testSortOn

    -- Subsequences
    print testTails
    print testInits
    print testChunksOf

    -- Filter
    print testTakeWhileL
    print testTakeWhileR
    print testDropWhileL
    print testDropWhileR
    print testSpanl
    print testBreakl
    print testPartition
    print testFilter

    -- Zip
    print testZip
    print testZipWith
    print testZipDifferentLengths
    print testUnzip

    -- Fold
    print testFoldMapWithIndex
    print testFoldlWithIndex
    print testFoldrWithIndex

    -- Map
    print testMapWithIndex

    -- Edge cases
    print testEmptyOperations
    print testSingleElement
    print testLargeSequence

    -- Properties
    print propFromToList
    print propConcatAssoc
    print propConcatEmpty
    print propReverseReverse
    print propTakeDropSplit
    print propConsLength
