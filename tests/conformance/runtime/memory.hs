-- Test: memory-management
-- Category: runtime
-- Profile: numeric
-- Expected: success
-- Spec: H26-SPEC Section 9

{-# PROFILE Numeric #-}

module MemoryTest where

import H26.Tensor
import H26.FFI

-- Pinned memory for FFI
test1 :: IO ()
test1 = do
  -- Allocate pinned memory (won't be moved by GC)
  ptr <- mallocPinnedBytes 1024
  -- Use with foreign function
  c_process_buffer ptr 1024
  -- Free when done
  free ptr

foreign import ccall "process_buffer"
  c_process_buffer :: Ptr () -> Int -> IO ()

-- Hot arena allocation (Numeric profile)
test2 :: Tensor Float -> Tensor Float
test2 xs = withArena $ \arena -> do
  -- Temporary buffer in arena (freed at scope end)
  tmp <- arenaAlloc arena (tensorSize xs)
  -- Process without GC allocation
  computeInPlace xs tmp
  result <- copy tmp
  pure result

-- Tensor memory layout
test3 :: Bool
test3 =
  let t = zeros [3, 4] :: Tensor Float
      -- Shape: [3, 4]
      -- Strides: [4, 1] (row-major, contiguous)
  in isContiguous t  -- True

-- Non-contiguous view (slice doesn't copy)
test4 :: Bool
test4 =
  let t = zeros [10, 10] :: Tensor Float
      v = slice [2..5] [3..7] t  -- 4x5 view
      -- Shares memory with t
      -- Strides: [10, 1] (not contiguous in memory)
  in isContiguous v  -- False

-- Forcing contiguous allocation
test5 :: Bool
test5 =
  let t = zeros [10, 10] :: Tensor Float
      v = slice [2..5] [3..7] t
      c = asContiguous v  -- Copies to contiguous memory
  in isContiguous c  -- True

-- Memory alignment for SIMD
test6 :: IO Bool
test6 = do
  ptr <- mallocAligned 1024 64  -- 64-byte aligned for AVX-512
  let aligned = ptrToWordPtr ptr `mod` 64 == 0
  free ptr
  pure aligned

-- No allocation in tight loops (Numeric profile guarantee)
test7 :: Float
test7 =
  let xs = fromList [1..1000] :: Tensor Float
      -- This loop should NOT allocate on general heap
      result = sum (map (*2) xs)  -- Fused, no intermediate
  in result

-- Mock functions (to be implemented)
withArena :: (Arena -> IO a) -> a
withArena = undefined

arenaAlloc :: Arena -> Int -> IO (Ptr a)
arenaAlloc = undefined

computeInPlace :: Tensor a -> Ptr a -> IO ()
computeInPlace = undefined

copy :: Ptr a -> IO (Tensor a)
copy = undefined

tensorSize :: Tensor a -> Int
tensorSize = undefined

isContiguous :: Tensor a -> Bool
isContiguous = undefined

asContiguous :: Tensor a -> Tensor a
asContiguous = undefined

ptrToWordPtr :: Ptr a -> Int
ptrToWordPtr = undefined

data Arena
