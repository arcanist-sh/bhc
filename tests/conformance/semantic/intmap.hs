-- Test: intmap
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Phase 3 - Containers

{-# HASKELL_EDITION 2026 #-}

module IntMapTest where

import qualified BHC.Data.IntMap as IM

-- ================================================================
-- Construction Tests
-- ================================================================

testEmpty :: Bool
testEmpty = IM.null IM.empty
-- Result: True

testSingleton :: Bool
testSingleton = IM.lookup 1 (IM.singleton 1 "one") == Just "one"
-- Result: True

testFromList :: Bool
testFromList =
    let m = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
    in IM.size m == 3
-- Result: True

testFromListWithDuplicates :: Bool
testFromListWithDuplicates =
    let m = IM.fromList [(1, "first"), (1, "second")]
    in IM.lookup 1 m == Just "second"  -- Last value wins
-- Result: True

-- ================================================================
-- Query Tests
-- ================================================================

testLookupPresent :: Bool
testLookupPresent =
    let m = IM.fromList [(1, "one"), (2, "two")]
    in IM.lookup 1 m == Just "one"
-- Result: True

testLookupAbsent :: Bool
testLookupAbsent =
    let m = IM.fromList [(1, "one"), (2, "two")]
    in IM.lookup 3 m == Nothing
-- Result: True

testMember :: Bool
testMember =
    let m = IM.fromList [(1, "a"), (2, "b")]
    in IM.member 1 m && not (IM.member 3 m)
-- Result: True

testNotMember :: Bool
testNotMember =
    let m = IM.fromList [(1, "a")]
    in IM.notMember 2 m
-- Result: True

testFindWithDefault :: Bool
testFindWithDefault =
    let m = IM.fromList [(1, 10)]
    in IM.findWithDefault 0 1 m == 10 && IM.findWithDefault 0 2 m == 0
-- Result: True

testSize :: Bool
testSize =
    let m = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
    in IM.size m == 3 && IM.size IM.empty == 0
-- Result: True

-- ================================================================
-- Insert/Delete Tests
-- ================================================================

testInsert :: Bool
testInsert =
    let m1 = IM.empty
        m2 = IM.insert 1 "one" m1
        m3 = IM.insert 2 "two" m2
    in IM.size m3 == 2 && IM.lookup 1 m3 == Just "one"
-- Result: True

testInsertOverwrite :: Bool
testInsertOverwrite =
    let m1 = IM.singleton 1 "old"
        m2 = IM.insert 1 "new" m1
    in IM.lookup 1 m2 == Just "new" && IM.size m2 == 1
-- Result: True

testDelete :: Bool
testDelete =
    let m1 = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
        m2 = IM.delete 2 m1
    in IM.size m2 == 2 && IM.lookup 2 m2 == Nothing
-- Result: True

testDeleteAbsent :: Bool
testDeleteAbsent =
    let m = IM.fromList [(1, "a")]
        m' = IM.delete 99 m
    in IM.size m' == 1  -- Deleting non-existent key is no-op
-- Result: True

testAdjust :: Bool
testAdjust =
    let m1 = IM.fromList [(1, 10), (2, 20)]
        m2 = IM.adjust (* 2) 1 m1
    in IM.lookup 1 m2 == Just 20 && IM.lookup 2 m2 == Just 20
-- Result: True

testUpdate :: Bool
testUpdate =
    let m1 = IM.fromList [(1, 10), (2, 20)]
        m2 = IM.update (\x -> if x > 15 then Just (x * 2) else Nothing) 1 m1
        m3 = IM.update (\x -> if x > 15 then Just (x * 2) else Nothing) 2 m1
    in IM.lookup 1 m2 == Nothing && IM.lookup 2 m3 == Just 40
-- Result: True

-- ================================================================
-- Combine Tests
-- ================================================================

testUnion :: Bool
testUnion =
    let m1 = IM.fromList [(1, "a"), (2, "b")]
        m2 = IM.fromList [(2, "B"), (3, "c")]
        m3 = IM.union m1 m2
    in IM.size m3 == 3 && IM.lookup 2 m3 == Just "a"  -- Left-biased
-- Result: True

testUnionWith :: Bool
testUnionWith =
    let m1 = IM.fromList [(1, 10), (2, 20)]
        m2 = IM.fromList [(2, 200), (3, 300)]
        m3 = IM.unionWith (+) m1 m2
    in IM.lookup 2 m3 == Just 220
-- Result: True

testIntersection :: Bool
testIntersection =
    let m1 = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
        m2 = IM.fromList [(2, "B"), (3, "C"), (4, "D")]
        m3 = IM.intersection m1 m2
    in IM.size m3 == 2 && IM.member 2 m3 && IM.member 3 m3
-- Result: True

testDifference :: Bool
testDifference =
    let m1 = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
        m2 = IM.fromList [(2, "B"), (4, "D")]
        m3 = IM.difference m1 m2
    in IM.size m3 == 2 && IM.member 1 m3 && IM.member 3 m3
-- Result: True

-- ================================================================
-- Traversal Tests
-- ================================================================

testMap :: Bool
testMap =
    let m1 = IM.fromList [(1, 10), (2, 20), (3, 30)]
        m2 = IM.map (* 2) m1
    in IM.lookup 1 m2 == Just 20 && IM.lookup 2 m2 == Just 40
-- Result: True

testMapWithKey :: Bool
testMapWithKey =
    let m1 = IM.fromList [(1, 10), (2, 20)]
        m2 = IM.mapWithKey (\k v -> k + v) m1
    in IM.lookup 1 m2 == Just 11 && IM.lookup 2 m2 == Just 22
-- Result: True

testFoldr :: Bool
testFoldr =
    let m = IM.fromList [(1, 10), (2, 20), (3, 30)]
    in IM.foldr (+) 0 m == 60
-- Result: True

testFoldl :: Bool
testFoldl =
    let m = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
    in IM.foldl (\acc x -> acc ++ x) "" m == "abc"  -- Order may vary
-- Result: True (order depends on key ordering)

testFoldrWithKey :: Bool
testFoldrWithKey =
    let m = IM.fromList [(1, 10), (2, 20)]
    in IM.foldrWithKey (\k v acc -> k + v + acc) 0 m == 33
-- Result: True (1 + 10 + 2 + 20 + 0)

-- ================================================================
-- Conversion Tests
-- ================================================================

testElems :: Bool
testElems =
    let m = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
        es = IM.elems m
    in length es == 3
-- Result: True

testKeys :: Bool
testKeys =
    let m = IM.fromList [(3, "c"), (1, "a"), (2, "b")]
        ks = IM.keys m
    in ks == [1, 2, 3]  -- Keys in ascending order
-- Result: True

testAssocs :: Bool
testAssocs =
    let m = IM.fromList [(2, "b"), (1, "a")]
        as = IM.assocs m
    in as == [(1, "a"), (2, "b")]  -- Sorted by key
-- Result: True

testToList :: Bool
testToList =
    let m = IM.fromList [(1, "a"), (2, "b")]
    in IM.toList m == [(1, "a"), (2, "b")]
-- Result: True

-- ================================================================
-- Filter Tests
-- ================================================================

testFilter :: Bool
testFilter =
    let m1 = IM.fromList [(1, 10), (2, 20), (3, 30), (4, 40)]
        m2 = IM.filter (> 20) m1
    in IM.size m2 == 2 && IM.member 3 m2 && IM.member 4 m2
-- Result: True

testFilterWithKey :: Bool
testFilterWithKey =
    let m1 = IM.fromList [(1, 10), (2, 20), (3, 30)]
        m2 = IM.filterWithKey (\k v -> k + v > 25) m1
    in IM.size m2 == 2  -- (2, 20) and (3, 30) pass
-- Result: True

testPartition :: Bool
testPartition =
    let m = IM.fromList [(1, 10), (2, 20), (3, 30)]
        (yes, no) = IM.partition (> 15) m
    in IM.size yes == 2 && IM.size no == 1
-- Result: True

-- ================================================================
-- Min/Max Tests
-- ================================================================

testFindMin :: Bool
testFindMin =
    let m = IM.fromList [(3, "c"), (1, "a"), (2, "b")]
    in IM.findMin m == (1, "a")
-- Result: True

testFindMax :: Bool
testFindMax =
    let m = IM.fromList [(1, "a"), (3, "c"), (2, "b")]
    in IM.findMax m == (3, "c")
-- Result: True

testDeleteMin :: Bool
testDeleteMin =
    let m1 = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
        m2 = IM.deleteMin m1
    in IM.size m2 == 2 && not (IM.member 1 m2)
-- Result: True

testDeleteMax :: Bool
testDeleteMax =
    let m1 = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
        m2 = IM.deleteMax m1
    in IM.size m2 == 2 && not (IM.member 3 m2)
-- Result: True

-- ================================================================
-- Split Tests
-- ================================================================

testSplit :: Bool
testSplit =
    let m = IM.fromList [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]
        (lt, gt) = IM.split 3 m
    in IM.size lt == 2 && IM.size gt == 2
-- Result: True (lt has 1,2; gt has 4,5)

testSplitLookup :: Bool
testSplitLookup =
    let m = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
        (lt, val, gt) = IM.splitLookup 2 m
    in IM.size lt == 1 && val == Just "b" && IM.size gt == 1
-- Result: True

-- ================================================================
-- Submap Tests
-- ================================================================

testIsSubmapOf :: Bool
testIsSubmapOf =
    let m1 = IM.fromList [(1, "a"), (2, "b")]
        m2 = IM.fromList [(1, "a"), (2, "b"), (3, "c")]
    in IM.isSubmapOf m1 m2 && not (IM.isSubmapOf m2 m1)
-- Result: True

-- ================================================================
-- Edge Cases
-- ================================================================

testEmptyOperations :: Bool
testEmptyOperations =
    let e = IM.empty
    in IM.null e &&
       IM.size e == 0 &&
       IM.lookup 1 e == Nothing &&
       IM.delete 1 e == e &&
       IM.toList e == []
-- Result: True

testNegativeKeys :: Bool
testNegativeKeys =
    let m = IM.fromList [(-1, "neg"), (0, "zero"), (1, "pos")]
    in IM.size m == 3 &&
       IM.lookup (-1) m == Just "neg" &&
       IM.lookup 0 m == Just "zero"
-- Result: True

testLargeKeys :: Bool
testLargeKeys =
    let m = IM.fromList [(1000000, "million"), (maxBound, "max")]
    in IM.size m == 2 && IM.member 1000000 m
-- Result: True

-- ================================================================
-- Property-style Tests
-- ================================================================

-- Insert then lookup returns the value
propInsertLookup :: Bool
propInsertLookup =
    let m = IM.insert 42 "answer" IM.empty
    in IM.lookup 42 m == Just "answer"
-- Result: True

-- Delete then lookup returns Nothing
propDeleteLookup :: Bool
propDeleteLookup =
    let m1 = IM.singleton 1 "one"
        m2 = IM.delete 1 m1
    in IM.lookup 1 m2 == Nothing
-- Result: True

-- fromList then toList preserves elements (sorted)
propFromToList :: Bool
propFromToList =
    let xs = [(3, "c"), (1, "a"), (2, "b")]
        m = IM.fromList xs
    in IM.toList m == [(1, "a"), (2, "b"), (3, "c")]
-- Result: True

-- Union is associative
propUnionAssoc :: Bool
propUnionAssoc =
    let a = IM.fromList [(1, "a")]
        b = IM.fromList [(2, "b")]
        c = IM.fromList [(3, "c")]
    in IM.union (IM.union a b) c == IM.union a (IM.union b c)
-- Result: True (comparing by toList)

-- ================================================================
-- Main
-- ================================================================

main :: IO ()
main = do
    -- Construction
    print testEmpty
    print testSingleton
    print testFromList
    print testFromListWithDuplicates

    -- Query
    print testLookupPresent
    print testLookupAbsent
    print testMember
    print testNotMember
    print testFindWithDefault
    print testSize

    -- Insert/Delete
    print testInsert
    print testInsertOverwrite
    print testDelete
    print testDeleteAbsent
    print testAdjust
    print testUpdate

    -- Combine
    print testUnion
    print testUnionWith
    print testIntersection
    print testDifference

    -- Traversal
    print testMap
    print testMapWithKey
    print testFoldr
    print testFoldl
    print testFoldrWithKey

    -- Conversion
    print testElems
    print testKeys
    print testAssocs
    print testToList

    -- Filter
    print testFilter
    print testFilterWithKey
    print testPartition

    -- Min/Max
    print testFindMin
    print testFindMax
    print testDeleteMin
    print testDeleteMax

    -- Split
    print testSplit
    print testSplitLookup

    -- Submap
    print testIsSubmapOf

    -- Edge cases
    print testEmptyOperations
    print testNegativeKeys
    print testLargeKeys

    -- Properties
    print propInsertLookup
    print propDeleteLookup
    print propFromToList
    print propUnionAssoc
