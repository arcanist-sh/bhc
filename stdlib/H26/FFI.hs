-- |
-- Module      : H26.FFI
-- Description : Foreign function interface
-- License     : BSD-3-Clause
--
-- The H26.FFI module provides safe foreign function interface bindings.
-- Supports calling C code, managing foreign memory, and marshalling
-- data across language boundaries.

{-# HASKELL_EDITION 2026 #-}

module H26.FFI
  ( -- * Foreign Pointers
    Ptr
  , FunPtr
  , nullPtr
  , nullFunPtr
  , castPtr
  , castFunPtr
  , plusPtr
  , minusPtr
  , alignPtr

    -- * Foreign References
  , ForeignPtr
  , newForeignPtr
  , newForeignPtr_
  , withForeignPtr
  , touchForeignPtr
  , castForeignPtr
  , mallocForeignPtr
  , mallocForeignPtrBytes
  , mallocForeignPtrArray
  , addForeignPtrFinalizer
  , finalizeForeignPtr

    -- * Storable Class
  , Storable(..)
  , sizeOf
  , alignment
  , peek
  , poke
  , peekByteOff
  , pokeByteOff
  , peekElemOff
  , pokeElemOff

    -- * Memory Allocation
  , malloc
  , mallocBytes
  , mallocArray
  , calloc
  , callocBytes
  , callocArray
  , realloc
  , reallocBytes
  , reallocArray
  , free
  , freeBytes

    -- * Pinned Allocation
  , mallocPinned
  , mallocPinnedBytes
  , mallocPinnedArray
  , mallocAligned
  , isPinned

    -- * Marshalling Utilities
  , alloca
  , allocaBytes
  , allocaArray
  , with
  , withArray
  , withArrayLen
  , withArray0
  , copyArray
  , moveArray
  , lengthArray0
  , advancePtr

    -- * Array Conversion
  , peekArray
  , peekArray0
  , pokeArray
  , pokeArray0
  , newArray
  , newArray0

    -- * String Marshalling
  , CString
  , CStringLen
  , peekCString
  , peekCStringLen
  , newCString
  , newCStringLen
  , withCString
  , withCStringLen
  , castCharToCChar
  , castCCharToChar

    -- * C Types
  , CChar
  , CSChar
  , CUChar
  , CShort
  , CUShort
  , CInt
  , CUInt
  , CLong
  , CULong
  , CLLong
  , CULLong
  , CFloat
  , CDouble
  , CSize
  , CSSize
  , CPtrdiff
  , CIntPtr
  , CUIntPtr
  , CBool

    -- * Stable Pointers
  , StablePtr
  , newStablePtr
  , deRefStablePtr
  , freeStablePtr
  , castStablePtrToPtr
  , castPtrToStablePtr

    -- * Function Pointers
  , FunPtrWrapper
  , wrapFunPtr
  , freeFunPtr
  , dynamicFunPtr

    -- * Error Handling
  , Errno(..)
  , getErrno
  , resetErrno
  , throwErrno
  , throwErrnoIf
  , throwErrnoIfMinus1
  , throwErrnoIfNull
  , throwErrnoPath
  , throwErrnoPathIf

    -- * Import/Export Utilities
  , CCallable
  , CReturnable
  , importC
  , exportC

    -- * Unsafe Operations
  , unsafePerformIO
  , unsafeInterleaveIO
  , unsafeDupablePerformIO
  , unsafeLocalState
  ) where

-- | Raw memory pointer.
data Ptr a

-- | Function pointer.
data FunPtr a

-- | Garbage-collected foreign pointer.
data ForeignPtr a

-- | Stable pointer (prevents GC of Haskell value).
data StablePtr a

-- | The null pointer.
nullPtr :: Ptr a

-- | The null function pointer.
nullFunPtr :: FunPtr a

-- | Cast pointer to different type.
castPtr :: Ptr a -> Ptr b

-- | Cast function pointer to different type.
castFunPtr :: FunPtr a -> FunPtr b

-- | Advance pointer by offset bytes.
plusPtr :: Ptr a -> Int -> Ptr b

-- | Compute byte difference between pointers.
minusPtr :: Ptr a -> Ptr b -> Int

-- | Align pointer to alignment boundary.
alignPtr :: Ptr a -> Int -> Ptr a

-- | Create foreign pointer with finalizer.
newForeignPtr :: FinalizerPtr a -> Ptr a -> IO (ForeignPtr a)

-- | Create foreign pointer without finalizer.
newForeignPtr_ :: Ptr a -> IO (ForeignPtr a)

-- | Execute action with foreign pointer.
withForeignPtr :: ForeignPtr a -> (Ptr a -> IO b) -> IO b

-- | Keep foreign pointer alive.
touchForeignPtr :: ForeignPtr a -> IO ()

-- | Cast foreign pointer to different type.
castForeignPtr :: ForeignPtr a -> ForeignPtr b

-- | Allocate foreign memory for Storable value.
mallocForeignPtr :: Storable a => IO (ForeignPtr a)

-- | Allocate foreign memory of given size.
mallocForeignPtrBytes :: Int -> IO (ForeignPtr a)

-- | Allocate foreign array.
mallocForeignPtrArray :: Storable a => Int -> IO (ForeignPtr a)

-- | Add finalizer to foreign pointer.
addForeignPtrFinalizer :: FinalizerPtr a -> ForeignPtr a -> IO ()

-- | Run finalizers immediately.
finalizeForeignPtr :: ForeignPtr a -> IO ()

-- | Class for types that can be stored in memory.
class Storable a where
  -- | Size of value in bytes.
  sizeOf :: a -> Int

  -- | Alignment requirement in bytes.
  alignment :: a -> Int

  -- | Read value from memory.
  peek :: Ptr a -> IO a

  -- | Write value to memory.
  poke :: Ptr a -> a -> IO ()

  -- | Read from byte offset.
  peekByteOff :: Ptr b -> Int -> IO a

  -- | Write at byte offset.
  pokeByteOff :: Ptr b -> Int -> a -> IO ()

  -- | Read from element offset.
  peekElemOff :: Ptr a -> Int -> IO a

  -- | Write at element offset.
  pokeElemOff :: Ptr a -> Int -> a -> IO ()

-- | Allocate memory for a Storable value.
malloc :: Storable a => IO (Ptr a)

-- | Allocate memory of given size.
mallocBytes :: Int -> IO (Ptr a)

-- | Allocate array of Storable values.
mallocArray :: Storable a => Int -> IO (Ptr a)

-- | Allocate zero-initialized memory.
calloc :: Storable a => IO (Ptr a)

-- | Allocate zero-initialized bytes.
callocBytes :: Int -> IO (Ptr a)

-- | Allocate zero-initialized array.
callocArray :: Storable a => Int -> IO (Ptr a)

-- | Resize allocated memory.
realloc :: Storable b => Ptr a -> IO (Ptr b)

-- | Resize to given size.
reallocBytes :: Ptr a -> Int -> IO (Ptr a)

-- | Resize array.
reallocArray :: Storable a => Ptr a -> Int -> IO (Ptr a)

-- | Free allocated memory.
free :: Ptr a -> IO ()

-- | Free allocated bytes.
freeBytes :: Ptr a -> IO ()

-- | Allocate pinned memory (won't be moved by GC).
mallocPinned :: Storable a => IO (Ptr a)

-- | Allocate pinned bytes.
mallocPinnedBytes :: Int -> IO (Ptr a)

-- | Allocate pinned array.
mallocPinnedArray :: Storable a => Int -> IO (Ptr a)

-- | Allocate with specific alignment.
mallocAligned :: Int -> Int -> IO (Ptr a)

-- | Check if memory is pinned.
isPinned :: Ptr a -> IO Bool

-- | Allocate temporary memory for Storable.
alloca :: Storable a => (Ptr a -> IO b) -> IO b

-- | Allocate temporary bytes.
allocaBytes :: Int -> (Ptr a -> IO b) -> IO b

-- | Allocate temporary array.
allocaArray :: Storable a => Int -> (Ptr a -> IO b) -> IO b

-- | Allocate and initialize with value.
with :: Storable a => a -> (Ptr a -> IO b) -> IO b

-- | Allocate and initialize array.
withArray :: Storable a => [a] -> (Ptr a -> IO b) -> IO b

-- | Allocate array with length.
withArrayLen :: Storable a => [a] -> (Int -> Ptr a -> IO b) -> IO b

-- | Allocate null-terminated array.
withArray0 :: Storable a => a -> [a] -> (Ptr a -> IO b) -> IO b

-- | Copy array elements.
copyArray :: Storable a => Ptr a -> Ptr a -> Int -> IO ()

-- | Move array elements (may overlap).
moveArray :: Storable a => Ptr a -> Ptr a -> Int -> IO ()

-- | Find length of null-terminated array.
lengthArray0 :: (Storable a, Eq a) => a -> Ptr a -> IO Int

-- | Advance pointer by element count.
advancePtr :: Storable a => Ptr a -> Int -> Ptr a

-- | Read array from memory.
peekArray :: Storable a => Int -> Ptr a -> IO [a]

-- | Read null-terminated array.
peekArray0 :: (Storable a, Eq a) => a -> Ptr a -> IO [a]

-- | Write array to memory.
pokeArray :: Storable a => Ptr a -> [a] -> IO ()

-- | Write null-terminated array.
pokeArray0 :: Storable a => a -> Ptr a -> [a] -> IO ()

-- | Allocate and initialize new array.
newArray :: Storable a => [a] -> IO (Ptr a)

-- | Allocate null-terminated array.
newArray0 :: Storable a => a -> [a] -> IO (Ptr a)

-- | C string (null-terminated).
type CString = Ptr CChar

-- | C string with length.
type CStringLen = (Ptr CChar, Int)

-- | Read C string to Haskell String.
peekCString :: CString -> IO String

-- | Read C string with length.
peekCStringLen :: CStringLen -> IO String

-- | Create new C string.
newCString :: String -> IO CString

-- | Create C string with length.
newCStringLen :: String -> IO CStringLen

-- | Execute with temporary C string.
withCString :: String -> (CString -> IO a) -> IO a

-- | Execute with temporary C string and length.
withCStringLen :: String -> (CStringLen -> IO a) -> IO a

-- | Convert Char to CChar.
castCharToCChar :: Char -> CChar

-- | Convert CChar to Char.
castCCharToChar :: CChar -> Char

-- | C signed char.
newtype CChar = CChar Int8

-- | C signed char (explicit).
newtype CSChar = CSChar Int8

-- | C unsigned char.
newtype CUChar = CUChar Word8

-- | C short.
newtype CShort = CShort Int16

-- | C unsigned short.
newtype CUShort = CUShort Word16

-- | C int.
newtype CInt = CInt Int32

-- | C unsigned int.
newtype CUInt = CUInt Word32

-- | C long.
newtype CLong = CLong Int64

-- | C unsigned long.
newtype CULong = CULong Word64

-- | C long long.
newtype CLLong = CLLong Int64

-- | C unsigned long long.
newtype CULLong = CULLong Word64

-- | C float.
newtype CFloat = CFloat Float

-- | C double.
newtype CDouble = CDouble Double

-- | C size_t.
newtype CSize = CSize Word64

-- | C ssize_t.
newtype CSSize = CSSize Int64

-- | C ptrdiff_t.
newtype CPtrdiff = CPtrdiff Int64

-- | C intptr_t.
newtype CIntPtr = CIntPtr Int64

-- | C uintptr_t.
newtype CUIntPtr = CUIntPtr Word64

-- | C bool.
newtype CBool = CBool Word8

-- | Create stable pointer.
newStablePtr :: a -> IO (StablePtr a)

-- | Dereference stable pointer.
deRefStablePtr :: StablePtr a -> IO a

-- | Free stable pointer.
freeStablePtr :: StablePtr a -> IO ()

-- | Cast stable pointer to raw pointer.
castStablePtrToPtr :: StablePtr a -> Ptr ()

-- | Cast raw pointer to stable pointer.
castPtrToStablePtr :: Ptr () -> StablePtr a

-- | Wrapper for exportable function pointers.
data FunPtrWrapper a

-- | Create function pointer from Haskell function.
wrapFunPtr :: a -> IO (FunPtr a)

-- | Free function pointer wrapper.
freeFunPtr :: FunPtr a -> IO ()

-- | Call function pointer dynamically.
dynamicFunPtr :: FunPtr a -> a

-- | C errno value.
newtype Errno = Errno CInt
  deriving (Eq, Show)

-- | Get current errno.
getErrno :: IO Errno

-- | Reset errno to zero.
resetErrno :: IO ()

-- | Throw IO error based on errno.
throwErrno :: String -> IO a

-- | Throw if predicate holds.
throwErrnoIf :: (a -> Bool) -> String -> IO a -> IO a

-- | Throw if result is -1.
throwErrnoIfMinus1 :: (Eq a, Num a) => String -> IO a -> IO a

-- | Throw if result is null.
throwErrnoIfNull :: String -> IO (Ptr a) -> IO (Ptr a)

-- | Throw with path information.
throwErrnoPath :: String -> FilePath -> IO a

-- | Throw with path if predicate holds.
throwErrnoPathIf :: (a -> Bool) -> String -> FilePath -> IO a -> IO a

-- | Constraint for types callable from C.
class CCallable a

-- | Constraint for types returnable to C.
class CReturnable a

-- | Import C function (compile-time).
--
-- Usage: foreign import ccall "function_name" ...
importC :: String -> a

-- | Export Haskell function to C (compile-time).
--
-- Usage: foreign export ccall "function_name" ...
exportC :: String -> a

-- | Perform IO unsafely.
--
-- __Warning__: Only use when you can guarantee:
-- - The IO action is pure (same result every time)
-- - The result doesn't depend on evaluation order
-- - No observable side effects
unsafePerformIO :: IO a -> a

-- | Interleave IO lazily.
--
-- The IO action runs when the result is demanded.
unsafeInterleaveIO :: IO a -> IO a

-- | Perform IO unsafely (may be duplicated).
--
-- Like unsafePerformIO but may run multiple times.
unsafeDupablePerformIO :: IO a -> a

-- | Create local mutable state.
unsafeLocalState :: IO a -> a

-- Internal type
type FinalizerPtr a = FunPtr (Ptr a -> IO ())

-- This is a specification file.
-- Actual implementation provided by the compiler.
