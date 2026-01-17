-- Test: ffi-interface
-- Category: runtime
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 11

{-# HASKELL_EDITION 2026 #-}

module FFITest where

import H26.FFI

-- Basic foreign import (safe)
foreign import ccall safe "strlen"
  c_strlen :: CString -> IO CSize

test1 :: IO Int
test1 = withCString "hello" $ \s -> do
  len <- c_strlen s
  pure (fromIntegral len)
-- Result: 5

-- Unsafe foreign import (fast but can't callback)
foreign import ccall unsafe "abs"
  c_abs :: CInt -> CInt

test2 :: Int
test2 = fromIntegral (c_abs (-42))
-- Result: 42

-- Foreign export
foreign export ccall "haskell_add"
  haskellAdd :: CInt -> CInt -> CInt

haskellAdd :: CInt -> CInt -> CInt
haskellAdd x y = x + y

-- Marshalling arrays
test3 :: IO [Int]
test3 = withArrayLen [1, 2, 3, 4, 5] $ \len ptr -> do
  result <- peekArray len ptr
  pure (map fromIntegral result)
-- Result: [1, 2, 3, 4, 5]

-- Storable instance for custom types
data Point = Point CDouble CDouble

instance Storable Point where
  sizeOf _ = 16
  alignment _ = 8
  peek ptr = Point
    <$> peekByteOff ptr 0
    <*> peekByteOff ptr 8
  poke ptr (Point x y) = do
    pokeByteOff ptr 0 x
    pokeByteOff ptr 8 y

test4 :: IO Point
test4 = alloca $ \ptr -> do
  poke ptr (Point 1.0 2.0)
  peek ptr
-- Result: Point 1.0 2.0

-- Foreign pointer with finalizer
test5 :: IO ()
test5 = do
  ptr <- mallocBytes 1024
  fptr <- newForeignPtr finalizerFree ptr
  withForeignPtr fptr $ \p -> do
    pokeByteOff p 0 (42 :: CInt)
  -- Memory freed when fptr is GC'd

foreign import ccall "&free"
  finalizerFree :: FinalizerPtr a

-- Stable pointers (prevent GC)
test6 :: IO Int
test6 = do
  let haskellValue = [1, 2, 3]
  sptr <- newStablePtr haskellValue
  -- Can now pass sptr to C code
  result <- deRefStablePtr sptr
  freeStablePtr sptr
  pure (sum result)
-- Result: 6

-- Function pointers
foreign import ccall "wrapper"
  mkCallback :: (CInt -> IO CInt) -> IO (FunPtr (CInt -> IO CInt))

test7 :: IO Int
test7 = do
  callback <- mkCallback (\x -> pure (x * 2))
  result <- invokeCallback callback 21
  freeHaskellFunPtr callback
  pure (fromIntegral result)
-- Result: 42

foreign import ccall "dynamic"
  invokeCallback :: FunPtr (CInt -> IO CInt) -> CInt -> IO CInt

-- Pinned memory requirement
test8 :: IO ()
test8 = do
  -- This memory is pinned (won't be moved by GC)
  ptr <- mallocPinnedBytes 1024
  -- Safe to use with async FFI operations
  asyncFFIOperation ptr
  -- Must explicitly free
  free ptr

foreign import ccall safe "async_ffi_operation"
  asyncFFIOperation :: Ptr () -> IO ()

-- CString conversion
test9 :: IO String
test9 = do
  cstr <- newCString "Hello, World!"
  result <- peekCString cstr
  free cstr
  pure result
-- Result: "Hello, World!"

-- Error handling with errno
foreign import ccall safe "open"
  c_open :: CString -> CInt -> IO CInt

test10 :: IO (Either Int Int)
test10 = withCString "/nonexistent" $ \path -> do
  resetErrno
  fd <- c_open path 0  -- O_RDONLY
  if fd < 0
    then do
      errno <- getErrno
      pure (Left (fromIntegral (unErrno errno)))
    else
      pure (Right (fromIntegral fd))
-- Result: Left ENOENT (or similar)

unErrno :: Errno -> CInt
unErrno (Errno n) = n

-- Type aliases for clarity
type CInt' = Int32
type CDouble' = Double
type CSize' = Word64

-- Mock function
freeHaskellFunPtr :: FunPtr a -> IO ()
freeHaskellFunPtr = undefined
