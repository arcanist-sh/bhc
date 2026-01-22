-- Test: custom-adt
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 4.2

{-# HASKELL_EDITION 2026 #-}

module CustomADTTest where

-- ================================================================
-- Simple Enumerations (no fields)
-- ================================================================

-- Simple 3-value enum
data Color = Red | Green | Blue

colorToInt :: Color -> Int
colorToInt c = case c of
  Red   -> 1
  Green -> 2
  Blue  -> 3

testColorRed :: Int
testColorRed = colorToInt Red
-- Result: 1

testColorGreen :: Int
testColorGreen = colorToInt Green
-- Result: 2

testColorBlue :: Int
testColorBlue = colorToInt Blue
-- Result: 3

-- Four-value enum
data Direction = North | East | South | West

directionToInt :: Direction -> Int
directionToInt d = case d of
  North -> 0
  East  -> 1
  South -> 2
  West  -> 3

testNorth :: Int
testNorth = directionToInt North
-- Result: 0

testWest :: Int
testWest = directionToInt West
-- Result: 3

-- ================================================================
-- Single-Constructor Types
-- ================================================================

-- Wrapper type
data Wrapper a = Wrap a

unwrap :: Wrapper a -> a
unwrap w = case w of
  Wrap x -> x

testWrapInt :: Int
testWrapInt = unwrap (Wrap 42)
-- Result: 42

testWrapBool :: Bool
testWrapBool = unwrap (Wrap True)
-- Result: True

-- Pair type
data Pair a b = MkPair a b

fstPair :: Pair a b -> a
fstPair p = case p of
  MkPair x _ -> x

sndPair :: Pair a b -> b
sndPair p = case p of
  MkPair _ y -> y

testPairFst :: Int
testPairFst = fstPair (MkPair 10 20)
-- Result: 10

testPairSnd :: Int
testPairSnd = sndPair (MkPair 10 20)
-- Result: 20

-- ================================================================
-- Multiple Constructors with Fields
-- ================================================================

-- Shape with different constructors
data Shape
  = Circle Int           -- radius
  | Rectangle Int Int    -- width, height
  | Triangle Int Int Int -- sides

area :: Shape -> Int
area s = case s of
  Circle r      -> 3 * r * r  -- simplified pi * r^2
  Rectangle w h -> w * h
  Triangle a b c ->
    -- Simplified: just return perimeter for testing
    a + b + c

testCircleArea :: Int
testCircleArea = area (Circle 5)
-- Result: 75 (3 * 25)

testRectArea :: Int
testRectArea = area (Rectangle 4 6)
-- Result: 24

testTriangleArea :: Int
testTriangleArea = area (Triangle 3 4 5)
-- Result: 12 (perimeter)

-- ================================================================
-- Recursive Types (Binary Tree)
-- ================================================================

data Tree a = Leaf a | Branch (Tree a) (Tree a)

-- Count leaves
countLeaves :: Tree a -> Int
countLeaves t = case t of
  Leaf _     -> 1
  Branch l r -> countLeaves l + countLeaves r

testLeafCount :: Int
testLeafCount = countLeaves (Leaf 42)
-- Result: 1

testBranchCount :: Int
testBranchCount = countLeaves (Branch (Leaf 1) (Leaf 2))
-- Result: 2

testDeepCount :: Int
testDeepCount = countLeaves (Branch (Branch (Leaf 1) (Leaf 2)) (Leaf 3))
-- Result: 3

-- Sum all values in tree
sumTree :: Tree Int -> Int
sumTree t = case t of
  Leaf n     -> n
  Branch l r -> sumTree l + sumTree r

testTreeSum :: Int
testTreeSum = sumTree (Branch (Leaf 10) (Branch (Leaf 20) (Leaf 30)))
-- Result: 60

-- Tree depth
depth :: Tree a -> Int
depth t = case t of
  Leaf _     -> 1
  Branch l r -> 1 + max (depth l) (depth r)

max :: Int -> Int -> Int
max x y = if x > y then x else y

testTreeDepth :: Int
testTreeDepth = depth (Branch (Branch (Leaf 1) (Leaf 2)) (Leaf 3))
-- Result: 3

-- ================================================================
-- Polymorphic Types
-- ================================================================

-- Option type (like Maybe but user-defined)
data Option a = None | Some a

fromOption :: a -> Option a -> a
fromOption def opt = case opt of
  None   -> def
  Some x -> x

testOptionNone :: Int
testOptionNone = fromOption 0 None
-- Result: 0

testOptionSome :: Int
testOptionSome = fromOption 0 (Some 42)
-- Result: 42

-- Result type (like Either but user-defined)
data Result e a = Err e | Ok a

isOk :: Result e a -> Bool
isOk r = case r of
  Err _ -> False
  Ok _  -> True

testResultErr :: Bool
testResultErr = isOk (Err "error")
-- Result: False

testResultOk :: Bool
testResultOk = isOk (Ok 42)
-- Result: True

-- Get value or default
getOrDefault :: a -> Result e a -> a
getOrDefault def r = case r of
  Err _ -> def
  Ok x  -> x

testGetErr :: Int
testGetErr = getOrDefault 0 (Err "error")
-- Result: 0

testGetOk :: Int
testGetOk = getOrDefault 0 (Ok 100)
-- Result: 100

-- ================================================================
-- Nested ADTs
-- ================================================================

-- Tree with optional values
data OptTree a = OptLeaf (Option a) | OptBranch (OptTree a) (OptTree a)

countSomes :: OptTree a -> Int
countSomes t = case t of
  OptLeaf opt -> case opt of
    None   -> 0
    Some _ -> 1
  OptBranch l r -> countSomes l + countSomes r

testCountSomes :: Int
testCountSomes = countSomes (OptBranch (OptLeaf (Some 1)) (OptLeaf None))
-- Result: 1

-- ================================================================
-- Pattern Matching Edge Cases
-- ================================================================

-- Constructor with same field types
data ThreeInts = ThreeInts Int Int Int

sumThree :: ThreeInts -> Int
sumThree t = case t of
  ThreeInts a b c -> a + b + c

testThreeInts :: Int
testThreeInts = sumThree (ThreeInts 10 20 30)
-- Result: 60

-- Nested pattern matching
matchNested :: Tree (Option Int) -> Int
matchNested t = case t of
  Leaf opt -> case opt of
    None   -> 0
    Some n -> n
  Branch l r -> matchNested l + matchNested r

testNestedMatch :: Int
testNestedMatch = matchNested (Branch (Leaf (Some 5)) (Leaf (Some 10)))
-- Result: 15

-- ================================================================
-- Mixed with Builtins
-- ================================================================

-- Use custom type with builtin Maybe
wrapMaybe :: Maybe a -> Option a
wrapMaybe m = case m of
  Nothing -> None
  Just x  -> Some x

testWrapMaybeNothing :: Int
testWrapMaybeNothing = fromOption 0 (wrapMaybe Nothing)
-- Result: 0

testWrapMaybeJust :: Int
testWrapMaybeJust = fromOption 0 (wrapMaybe (Just 42))
-- Result: 42

-- Use custom type in list
sumOptions :: [Option Int] -> Int
sumOptions xs = case xs of
  []     -> 0
  (o:os) -> fromOption 0 o + sumOptions os

testSumOptions :: Int
testSumOptions = sumOptions [Some 10, None, Some 20, Some 30]
-- Result: 60

-- ================================================================
-- Construction and Deconstruction
-- ================================================================

-- Build tree from list
buildTree :: [a] -> Option (Tree a)
buildTree xs = case xs of
  []  -> None
  [x] -> Some (Leaf x)
  _   -> case splitList xs of
           MkPair l r -> case buildTree l of
             None -> None
             Some lt -> case buildTree r of
               None -> None
               Some rt -> Some (Branch lt rt)

-- Split list in half (simplified)
splitList :: [a] -> Pair [a] [a]
splitList xs = splitAt (length xs `div` 2) xs
  where
    splitAt :: Int -> [a] -> Pair [a] [a]
    splitAt n ys = case n of
      0 -> MkPair [] ys
      _ -> case ys of
        []     -> MkPair [] []
        (z:zs) -> case splitAt (n - 1) zs of
          MkPair left right -> MkPair (z : left) right

-- Test tree building
testBuildTree :: Int
testBuildTree = case buildTree [1, 2, 3, 4] of
  None   -> 0
  Some t -> sumTree t
-- Result: 10

-- ================================================================
-- Main function to run all tests
-- ================================================================

main :: IO ()
main = do
  -- Enumerations
  putStrLn "=== Enumeration tests ==="
  print testColorRed     -- Expected: 1
  print testColorGreen   -- Expected: 2
  print testColorBlue    -- Expected: 3
  print testNorth        -- Expected: 0
  print testWest         -- Expected: 3

  -- Single constructor
  putStrLn "=== Single constructor tests ==="
  print testWrapInt      -- Expected: 42
  print testPairFst      -- Expected: 10
  print testPairSnd      -- Expected: 20

  -- Multiple constructors
  putStrLn "=== Shape tests ==="
  print testCircleArea   -- Expected: 75
  print testRectArea     -- Expected: 24
  print testTriangleArea -- Expected: 12

  -- Recursive types
  putStrLn "=== Tree tests ==="
  print testLeafCount    -- Expected: 1
  print testBranchCount  -- Expected: 2
  print testDeepCount    -- Expected: 3
  print testTreeSum      -- Expected: 60
  print testTreeDepth    -- Expected: 3

  -- Polymorphic types
  putStrLn "=== Option/Result tests ==="
  print testOptionNone   -- Expected: 0
  print testOptionSome   -- Expected: 42
  print testResultErr    -- Expected: False (0)
  print testResultOk     -- Expected: True (1)
  print testGetErr       -- Expected: 0
  print testGetOk        -- Expected: 100

  -- Nested
  putStrLn "=== Nested tests ==="
  print testCountSomes   -- Expected: 1
  print testThreeInts    -- Expected: 60
  print testNestedMatch  -- Expected: 15

  -- Mixed with builtins
  putStrLn "=== Mixed tests ==="
  print testWrapMaybeNothing -- Expected: 0
  print testWrapMaybeJust    -- Expected: 42
  print testSumOptions       -- Expected: 60

  -- Tree building
  putStrLn "=== Tree building tests ==="
  print testBuildTree    -- Expected: 10

  putStrLn "=== All custom ADT tests completed ==="
