-- Test: intset
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Phase 3 - Containers

{-# HASKELL_EDITION 2026 #-}

module IntSetTest where

import qualified BHC.Data.IntSet as IS

-- ================================================================
-- Construction Tests
-- ================================================================

testEmpty :: Bool
testEmpty = IS.null IS.empty && IS.size IS.empty == 0
-- Result: True

testSingleton :: Bool
testSingleton =
    let s = IS.singleton 42
    in IS.size s == 1 && IS.member 42 s
-- Result: True

testFromList :: Bool
testFromList =
    let s = IS.fromList [1, 2, 3, 4, 5]
    in IS.size s == 5
-- Result: True

testFromListWithDuplicates :: Bool
testFromListWithDuplicates =
    let s = IS.fromList [1, 1, 2, 2, 3, 3]
    in IS.size s == 3  -- Duplicates removed
-- Result: True

testFromAscList :: Bool
testFromAscList =
    let s = IS.fromAscList [1, 2, 3, 4, 5]
    in IS.size s == 5 && IS.toAscList s == [1, 2, 3, 4, 5]
-- Result: True

testFromDistinctAscList :: Bool
testFromDistinctAscList =
    let s = IS.fromDistinctAscList [10, 20, 30]
    in IS.size s == 3
-- Result: True

-- ================================================================
-- Query Tests
-- ================================================================

testMember :: Bool
testMember =
    let s = IS.fromList [1, 2, 3]
    in IS.member 2 s && not (IS.member 4 s)
-- Result: True

testNotMember :: Bool
testNotMember =
    let s = IS.fromList [1, 2, 3]
    in IS.notMember 4 s && not (IS.notMember 2 s)
-- Result: True

testLookupLT :: Bool
testLookupLT =
    let s = IS.fromList [1, 3, 5, 7, 9]
    in IS.lookupLT 5 s == Just 3 && IS.lookupLT 1 s == Nothing
-- Result: True

testLookupGT :: Bool
testLookupGT =
    let s = IS.fromList [1, 3, 5, 7, 9]
    in IS.lookupGT 5 s == Just 7 && IS.lookupGT 9 s == Nothing
-- Result: True

testLookupLE :: Bool
testLookupLE =
    let s = IS.fromList [1, 3, 5, 7, 9]
    in IS.lookupLE 5 s == Just 5 && IS.lookupLE 4 s == Just 3
-- Result: True

testLookupGE :: Bool
testLookupGE =
    let s = IS.fromList [1, 3, 5, 7, 9]
    in IS.lookupGE 5 s == Just 5 && IS.lookupGE 6 s == Just 7
-- Result: True

testSize :: Bool
testSize =
    IS.size IS.empty == 0 &&
    IS.size (IS.singleton 1) == 1 &&
    IS.size (IS.fromList [1, 2, 3]) == 3
-- Result: True

testNull :: Bool
testNull =
    IS.null IS.empty && not (IS.null (IS.singleton 1))
-- Result: True

-- ================================================================
-- Insert/Delete Tests
-- ================================================================

testInsert :: Bool
testInsert =
    let s1 = IS.empty
        s2 = IS.insert 1 s1
        s3 = IS.insert 2 s2
        s4 = IS.insert 3 s3
    in IS.size s4 == 3 && IS.member 2 s4
-- Result: True

testInsertDuplicate :: Bool
testInsertDuplicate =
    let s1 = IS.singleton 1
        s2 = IS.insert 1 s1
    in IS.size s2 == 1  -- No change
-- Result: True

testDelete :: Bool
testDelete =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.delete 2 s1
    in IS.size s2 == 2 && not (IS.member 2 s2)
-- Result: True

testDeleteAbsent :: Bool
testDeleteAbsent =
    let s = IS.fromList [1, 2, 3]
        s' = IS.delete 99 s
    in IS.size s' == 3  -- No change
-- Result: True

-- ================================================================
-- Combine Tests
-- ================================================================

testUnion :: Bool
testUnion =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.fromList [3, 4, 5]
        s3 = IS.union s1 s2
    in IS.size s3 == 5 && IS.toAscList s3 == [1, 2, 3, 4, 5]
-- Result: True

testUnions :: Bool
testUnions =
    let sets = [IS.fromList [1, 2], IS.fromList [2, 3], IS.fromList [3, 4]]
        s = IS.unions sets
    in IS.size s == 4
-- Result: True

testIntersection :: Bool
testIntersection =
    let s1 = IS.fromList [1, 2, 3, 4]
        s2 = IS.fromList [3, 4, 5, 6]
        s3 = IS.intersection s1 s2
    in IS.size s3 == 2 && IS.toAscList s3 == [3, 4]
-- Result: True

testDifference :: Bool
testDifference =
    let s1 = IS.fromList [1, 2, 3, 4]
        s2 = IS.fromList [3, 4, 5, 6]
        s3 = IS.difference s1 s2
    in IS.size s3 == 2 && IS.toAscList s3 == [1, 2]
-- Result: True

testSymmetricDifference :: Bool
testSymmetricDifference =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.fromList [2, 3, 4]
        -- Symmetric difference: elements in exactly one set
        sym = IS.union (IS.difference s1 s2) (IS.difference s2 s1)
    in IS.toAscList sym == [1, 4]
-- Result: True

-- ================================================================
-- Filter Tests
-- ================================================================

testFilter :: Bool
testFilter =
    let s1 = IS.fromList [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        s2 = IS.filter even s1
    in IS.size s2 == 5 && IS.toAscList s2 == [2, 4, 6, 8, 10]
-- Result: True

testPartition :: Bool
testPartition =
    let s = IS.fromList [1, 2, 3, 4, 5]
        (evens, odds) = IS.partition even s
    in IS.size evens == 2 && IS.size odds == 3
-- Result: True

-- ================================================================
-- Map Tests
-- ================================================================

testMap :: Bool
testMap =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.map (* 2) s1
    in IS.toAscList s2 == [2, 4, 6]
-- Result: True

testMapCollision :: Bool
testMapCollision =
    let s1 = IS.fromList [1, 2, 3, 4]
        s2 = IS.map (`div` 2) s1  -- 1,2 -> 0,1 and 3,4 -> 1,2
    in IS.size s2 < IS.size s1  -- Some values collide
-- Result: True

-- ================================================================
-- Fold Tests
-- ================================================================

testFoldr :: Bool
testFoldr =
    let s = IS.fromList [1, 2, 3, 4, 5]
    in IS.foldr (+) 0 s == 15
-- Result: True

testFoldl :: Bool
testFoldl =
    let s = IS.fromList [1, 2, 3]
    in IS.foldl (+) 0 s == 6
-- Result: True

testFoldr' :: Bool
testFoldr' =
    let s = IS.fromList [1, 2, 3, 4, 5]
    in IS.foldr' (+) 0 s == 15
-- Result: True

testFoldl' :: Bool
testFoldl' =
    let s = IS.fromList [1, 2, 3]
    in IS.foldl' (+) 0 s == 6
-- Result: True

-- ================================================================
-- Min/Max Tests
-- ================================================================

testFindMin :: Bool
testFindMin =
    let s = IS.fromList [5, 3, 8, 1, 9]
    in IS.findMin s == 1
-- Result: True

testFindMax :: Bool
testFindMax =
    let s = IS.fromList [5, 3, 8, 1, 9]
    in IS.findMax s == 9
-- Result: True

testDeleteMin :: Bool
testDeleteMin =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.deleteMin s1
    in IS.size s2 == 2 && not (IS.member 1 s2)
-- Result: True

testDeleteMax :: Bool
testDeleteMax =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.deleteMax s1
    in IS.size s2 == 2 && not (IS.member 3 s2)
-- Result: True

testMinView :: Bool
testMinView =
    let s = IS.fromList [1, 2, 3]
    in case IS.minView s of
        Just (m, s') -> m == 1 && IS.size s' == 2
        Nothing -> False
-- Result: True

testMaxView :: Bool
testMaxView =
    let s = IS.fromList [1, 2, 3]
    in case IS.maxView s of
        Just (m, s') -> m == 3 && IS.size s' == 2
        Nothing -> False
-- Result: True

-- ================================================================
-- Conversion Tests
-- ================================================================

testToList :: Bool
testToList =
    let s = IS.fromList [3, 1, 4, 1, 5, 9, 2, 6]
    in IS.toList s == IS.toAscList s  -- toList returns ascending
-- Result: True

testToAscList :: Bool
testToAscList =
    let s = IS.fromList [5, 3, 1, 4, 2]
    in IS.toAscList s == [1, 2, 3, 4, 5]
-- Result: True

testToDescList :: Bool
testToDescList =
    let s = IS.fromList [1, 2, 3, 4, 5]
    in IS.toDescList s == [5, 4, 3, 2, 1]
-- Result: True

-- ================================================================
-- Subset Tests
-- ================================================================

testIsSubsetOf :: Bool
testIsSubsetOf =
    let s1 = IS.fromList [1, 2]
        s2 = IS.fromList [1, 2, 3]
    in IS.isSubsetOf s1 s2 && not (IS.isSubsetOf s2 s1)
-- Result: True

testIsProperSubsetOf :: Bool
testIsProperSubsetOf =
    let s1 = IS.fromList [1, 2]
        s2 = IS.fromList [1, 2, 3]
        s3 = IS.fromList [1, 2]
    in IS.isProperSubsetOf s1 s2 && not (IS.isProperSubsetOf s1 s3)
-- Result: True

testDisjoint :: Bool
testDisjoint =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.fromList [4, 5, 6]
        s3 = IS.fromList [3, 4, 5]
    in IS.disjoint s1 s2 && not (IS.disjoint s1 s3)
-- Result: True

-- ================================================================
-- Split Tests
-- ================================================================

testSplit :: Bool
testSplit =
    let s = IS.fromList [1, 2, 3, 4, 5]
        (lt, gt) = IS.split 3 s
    in IS.toAscList lt == [1, 2] && IS.toAscList gt == [4, 5]
-- Result: True

testSplitMember :: Bool
testSplitMember =
    let s = IS.fromList [1, 2, 3, 4, 5]
        (lt, found, gt) = IS.splitMember 3 s
    in found && IS.size lt == 2 && IS.size gt == 2
-- Result: True

testSplitMemberAbsent :: Bool
testSplitMemberAbsent =
    let s = IS.fromList [1, 2, 4, 5]
        (lt, found, gt) = IS.splitMember 3 s
    in not found && IS.size lt == 2 && IS.size gt == 2
-- Result: True

-- ================================================================
-- Edge Cases
-- ================================================================

testEmptyOperations :: Bool
testEmptyOperations =
    let e = IS.empty
    in IS.null e &&
       IS.size e == 0 &&
       not (IS.member 1 e) &&
       IS.delete 1 e == e &&
       IS.toList e == []
-- Result: True

testNegativeInts :: Bool
testNegativeInts =
    let s = IS.fromList [-5, -3, -1, 0, 1, 3, 5]
    in IS.size s == 7 &&
       IS.member (-3) s &&
       IS.findMin s == (-5) &&
       IS.findMax s == 5
-- Result: True

testLargeInts :: Bool
testLargeInts =
    let s = IS.fromList [1000000, 2000000, maxBound `div` 2]
    in IS.size s == 3 && IS.member 1000000 s
-- Result: True

testSingleElement :: Bool
testSingleElement =
    let s = IS.singleton 42
    in IS.size s == 1 &&
       IS.member 42 s &&
       IS.findMin s == 42 &&
       IS.findMax s == 42 &&
       IS.toList s == [42]
-- Result: True

-- ================================================================
-- Property-style Tests
-- ================================================================

-- Insert then member returns True
propInsertMember :: Bool
propInsertMember =
    let s = IS.insert 99 IS.empty
    in IS.member 99 s
-- Result: True

-- Delete then member returns False
propDeleteMember :: Bool
propDeleteMember =
    let s1 = IS.singleton 42
        s2 = IS.delete 42 s1
    in not (IS.member 42 s2)
-- Result: True

-- Union is commutative
propUnionCommutative :: Bool
propUnionCommutative =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.fromList [3, 4, 5]
    in IS.union s1 s2 == IS.union s2 s1
-- Result: True

-- Intersection is commutative
propIntersectionCommutative :: Bool
propIntersectionCommutative =
    let s1 = IS.fromList [1, 2, 3]
        s2 = IS.fromList [2, 3, 4]
    in IS.intersection s1 s2 == IS.intersection s2 s1
-- Result: True

-- Union with empty is identity
propUnionEmpty :: Bool
propUnionEmpty =
    let s = IS.fromList [1, 2, 3]
    in IS.union s IS.empty == s && IS.union IS.empty s == s
-- Result: True

-- Intersection with empty is empty
propIntersectionEmpty :: Bool
propIntersectionEmpty =
    let s = IS.fromList [1, 2, 3]
    in IS.null (IS.intersection s IS.empty)
-- Result: True

-- fromList then toList preserves elements
propFromToList :: Bool
propFromToList =
    let xs = [5, 2, 8, 1, 9, 3]
        s = IS.fromList xs
        ys = IS.toAscList s
    in ys == [1, 2, 3, 5, 8, 9]
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
    print testFromListWithDuplicates
    print testFromAscList
    print testFromDistinctAscList

    -- Query
    print testMember
    print testNotMember
    print testLookupLT
    print testLookupGT
    print testLookupLE
    print testLookupGE
    print testSize
    print testNull

    -- Insert/Delete
    print testInsert
    print testInsertDuplicate
    print testDelete
    print testDeleteAbsent

    -- Combine
    print testUnion
    print testUnions
    print testIntersection
    print testDifference
    print testSymmetricDifference

    -- Filter
    print testFilter
    print testPartition

    -- Map
    print testMap
    print testMapCollision

    -- Fold
    print testFoldr
    print testFoldl
    print testFoldr'
    print testFoldl'

    -- Min/Max
    print testFindMin
    print testFindMax
    print testDeleteMin
    print testDeleteMax
    print testMinView
    print testMaxView

    -- Conversion
    print testToList
    print testToAscList
    print testToDescList

    -- Subset
    print testIsSubsetOf
    print testIsProperSubsetOf
    print testDisjoint

    -- Split
    print testSplit
    print testSplitMember
    print testSplitMemberAbsent

    -- Edge cases
    print testEmptyOperations
    print testNegativeInts
    print testLargeInts
    print testSingleElement

    -- Properties
    print propInsertMember
    print propDeleteMember
    print propUnionCommutative
    print propIntersectionCommutative
    print propUnionEmpty
    print propIntersectionEmpty
    print propFromToList
