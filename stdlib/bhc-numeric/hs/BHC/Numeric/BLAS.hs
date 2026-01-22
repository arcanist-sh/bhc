-- |
-- Module      : BHC.Numeric.BLAS
-- Description : BLAS (Basic Linear Algebra Subprograms) bindings
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Bindings to BLAS for high-performance linear algebra.
-- Supports multiple backends: OpenBLAS, MKL, Accelerate, and pure BHC fallback.

{-# LANGUAGE ForeignFunctionInterface #-}

module BHC.Numeric.BLAS (
    -- * BLAS Provider
    BLASProvider(..),
    currentProvider,
    setProvider,

    -- * Available providers
    OpenBLAS(..),
    MKL(..),
    Accelerate(..),
    PureBHC(..),

    -- * Level 1 BLAS (Vector-Vector)
    -- ** Double precision
    ddot, dnrm2, dasum, idamax,
    dswap, dcopy, daxpy, dscal,
    drot, drotg, drotm, drotmg,

    -- ** Single precision
    sdot, snrm2, sasum, isamax,
    sswap, scopy, saxpy, sscal,

    -- * Level 2 BLAS (Matrix-Vector)
    -- ** Double precision
    dgemv, dtrmv, dtrsv,
    dger, dsyr, dsyr2,

    -- ** Single precision
    sgemv, strmv, strsv,
    sger, ssyr, ssyr2,

    -- * Level 3 BLAS (Matrix-Matrix)
    -- ** Double precision
    dgemm, dsymm, dsyrk, dsyr2k,
    dtrmm, dtrsm,

    -- ** Single precision
    sgemm, ssymm, ssyrk, ssyr2k,
    strmm, strsm,

    -- * Transpose and layout
    Transpose(..),
    Layout(..),
    UpLo(..),
    Diag(..),
    Side(..),
) where

import BHC.Prelude
import Foreign.Ptr

-- ============================================================
-- BLAS Provider
-- ============================================================

-- | Type class for BLAS implementations.
class BLASProvider p where
    -- | Name of the provider.
    providerName :: p -> String

    -- | Check if provider is available.
    isAvailable :: p -> IO Bool

    -- Level 1
    blasDdot  :: p -> Int -> Ptr Double -> Int -> Ptr Double -> Int -> IO Double
    blasDnrm2 :: p -> Int -> Ptr Double -> Int -> IO Double
    blasDaxpy :: p -> Int -> Double -> Ptr Double -> Int -> Ptr Double -> Int -> IO ()
    blasDscal :: p -> Int -> Double -> Ptr Double -> Int -> IO ()

    -- Level 2
    blasDgemv :: p -> Layout -> Transpose -> Int -> Int
              -> Double -> Ptr Double -> Int
              -> Ptr Double -> Int
              -> Double -> Ptr Double -> Int -> IO ()

    -- Level 3
    blasDgemm :: p -> Layout -> Transpose -> Transpose
              -> Int -> Int -> Int
              -> Double -> Ptr Double -> Int
              -> Ptr Double -> Int
              -> Double -> Ptr Double -> Int -> IO ()

-- | OpenBLAS provider.
data OpenBLAS = OpenBLAS

instance BLASProvider OpenBLAS where
    providerName _ = "OpenBLAS"
    isAvailable _ = c_openblas_available
    blasDdot _ = c_cblas_ddot
    blasDnrm2 _ = c_cblas_dnrm2
    blasDaxpy _ = c_cblas_daxpy
    blasDscal _ = c_cblas_dscal
    blasDgemv _ = c_cblas_dgemv
    blasDgemm _ = c_cblas_dgemm

-- | Intel MKL provider.
data MKL = MKL

instance BLASProvider MKL where
    providerName _ = "Intel MKL"
    isAvailable _ = c_mkl_available
    blasDdot _ = c_mkl_ddot
    blasDnrm2 _ = c_mkl_dnrm2
    blasDaxpy _ = c_mkl_daxpy
    blasDscal _ = c_mkl_dscal
    blasDgemv _ = c_mkl_dgemv
    blasDgemm _ = c_mkl_dgemm

-- | Apple Accelerate provider (macOS only).
data Accelerate = Accelerate

instance BLASProvider Accelerate where
    providerName _ = "Apple Accelerate"
    isAvailable _ = c_accelerate_available
    blasDdot _ = c_cblas_ddot  -- Accelerate uses cblas interface
    blasDnrm2 _ = c_cblas_dnrm2
    blasDaxpy _ = c_cblas_daxpy
    blasDscal _ = c_cblas_dscal
    blasDgemv _ = c_cblas_dgemv
    blasDgemm _ = c_cblas_dgemm

-- | Pure BHC fallback (no external dependencies).
data PureBHC = PureBHC

instance BLASProvider PureBHC where
    providerName _ = "Pure BHC"
    isAvailable _ = pure True  -- Always available
    blasDdot _ = c_bhc_ddot
    blasDnrm2 _ = c_bhc_dnrm2
    blasDaxpy _ = c_bhc_daxpy
    blasDscal _ = c_bhc_dscal
    blasDgemv _ = c_bhc_dgemv
    blasDgemm _ = c_bhc_dgemm

-- | Get current BLAS provider.
foreign import ccall "bhc_blas_get_provider"
    currentProvider :: IO String

-- | Set BLAS provider.
foreign import ccall "bhc_blas_set_provider"
    setProvider :: String -> IO Bool

-- ============================================================
-- Enumerations
-- ============================================================

-- | Matrix transpose operation.
data Transpose
    = NoTrans    -- ^ No transpose (A)
    | Trans      -- ^ Transpose (A^T)
    | ConjTrans  -- ^ Conjugate transpose (A^H)
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Matrix storage layout.
data Layout
    = RowMajor   -- ^ Row-major (C-style)
    | ColMajor   -- ^ Column-major (Fortran-style)
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Upper or lower triangular.
data UpLo
    = Upper
    | Lower
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Diagonal type.
data Diag
    = NonUnit    -- ^ General diagonal
    | Unit       -- ^ Unit diagonal (all 1s)
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Side for matrix multiplication.
data Side
    = Left_      -- ^ Multiply from left
    | Right_     -- ^ Multiply from right
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- ============================================================
-- Level 1 BLAS (Vector-Vector)
-- ============================================================

-- | Dot product of two double vectors.
--
-- @ddot n x incx y incy = sum(x[i] * y[i])@
foreign import ccall "cblas_ddot"
    ddot :: Int -> Ptr Double -> Int -> Ptr Double -> Int -> IO Double

-- | Euclidean norm of a double vector.
--
-- @dnrm2 n x incx = sqrt(sum(x[i]^2))@
foreign import ccall "cblas_dnrm2"
    dnrm2 :: Int -> Ptr Double -> Int -> IO Double

-- | Sum of absolute values.
--
-- @dasum n x incx = sum(|x[i]|)@
foreign import ccall "cblas_dasum"
    dasum :: Int -> Ptr Double -> Int -> IO Double

-- | Index of maximum absolute value.
foreign import ccall "cblas_idamax"
    idamax :: Int -> Ptr Double -> Int -> IO Int

-- | Swap two vectors.
foreign import ccall "cblas_dswap"
    dswap :: Int -> Ptr Double -> Int -> Ptr Double -> Int -> IO ()

-- | Copy vector.
foreign import ccall "cblas_dcopy"
    dcopy :: Int -> Ptr Double -> Int -> Ptr Double -> Int -> IO ()

-- | AXPY: y = alpha * x + y
--
-- @daxpy n alpha x incx y incy@ computes @y := alpha * x + y@
foreign import ccall "cblas_daxpy"
    daxpy :: Int -> Double -> Ptr Double -> Int -> Ptr Double -> Int -> IO ()

-- | Scale vector: x = alpha * x
foreign import ccall "cblas_dscal"
    dscal :: Int -> Double -> Ptr Double -> Int -> IO ()

-- | Apply rotation.
foreign import ccall "cblas_drot"
    drot :: Int -> Ptr Double -> Int -> Ptr Double -> Int -> Double -> Double -> IO ()

-- | Generate rotation.
foreign import ccall "cblas_drotg"
    drotg :: Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

-- | Apply modified rotation.
foreign import ccall "cblas_drotm"
    drotm :: Int -> Ptr Double -> Int -> Ptr Double -> Int -> Ptr Double -> IO ()

-- | Generate modified rotation.
foreign import ccall "cblas_drotmg"
    drotmg :: Ptr Double -> Ptr Double -> Ptr Double -> Double -> Ptr Double -> IO ()

-- Single precision versions
foreign import ccall "cblas_sdot"
    sdot :: Int -> Ptr Float -> Int -> Ptr Float -> Int -> IO Float

foreign import ccall "cblas_snrm2"
    snrm2 :: Int -> Ptr Float -> Int -> IO Float

foreign import ccall "cblas_sasum"
    sasum :: Int -> Ptr Float -> Int -> IO Float

foreign import ccall "cblas_isamax"
    isamax :: Int -> Ptr Float -> Int -> IO Int

foreign import ccall "cblas_sswap"
    sswap :: Int -> Ptr Float -> Int -> Ptr Float -> Int -> IO ()

foreign import ccall "cblas_scopy"
    scopy :: Int -> Ptr Float -> Int -> Ptr Float -> Int -> IO ()

foreign import ccall "cblas_saxpy"
    saxpy :: Int -> Float -> Ptr Float -> Int -> Ptr Float -> Int -> IO ()

foreign import ccall "cblas_sscal"
    sscal :: Int -> Float -> Ptr Float -> Int -> IO ()

-- ============================================================
-- Level 2 BLAS (Matrix-Vector)
-- ============================================================

-- | General matrix-vector multiplication.
--
-- @y := alpha * op(A) * x + beta * y@
--
-- where @op(A)@ is @A@, @A^T@, or @A^H@ depending on trans.
foreign import ccall "cblas_dgemv"
    dgemv :: Int -> Int  -- Layout, Trans
          -> Int -> Int  -- M, N
          -> Double -> Ptr Double -> Int  -- alpha, A, lda
          -> Ptr Double -> Int  -- x, incx
          -> Double -> Ptr Double -> Int  -- beta, y, incy
          -> IO ()

-- | Triangular matrix-vector multiply.
foreign import ccall "cblas_dtrmv"
    dtrmv :: Int -> Int -> Int -> Int  -- Layout, UpLo, Trans, Diag
          -> Int -> Ptr Double -> Int  -- N, A, lda
          -> Ptr Double -> Int  -- x, incx
          -> IO ()

-- | Triangular solve.
foreign import ccall "cblas_dtrsv"
    dtrsv :: Int -> Int -> Int -> Int  -- Layout, UpLo, Trans, Diag
          -> Int -> Ptr Double -> Int  -- N, A, lda
          -> Ptr Double -> Int  -- x, incx
          -> IO ()

-- | Rank-1 update: A := alpha * x * y^T + A
foreign import ccall "cblas_dger"
    dger :: Int  -- Layout
         -> Int -> Int  -- M, N
         -> Double -> Ptr Double -> Int  -- alpha, x, incx
         -> Ptr Double -> Int  -- y, incy
         -> Ptr Double -> Int  -- A, lda
         -> IO ()

-- | Symmetric rank-1 update.
foreign import ccall "cblas_dsyr"
    dsyr :: Int -> Int  -- Layout, UpLo
         -> Int  -- N
         -> Double -> Ptr Double -> Int  -- alpha, x, incx
         -> Ptr Double -> Int  -- A, lda
         -> IO ()

-- | Symmetric rank-2 update.
foreign import ccall "cblas_dsyr2"
    dsyr2 :: Int -> Int  -- Layout, UpLo
          -> Int  -- N
          -> Double -> Ptr Double -> Int  -- alpha, x, incx
          -> Ptr Double -> Int  -- y, incy
          -> Ptr Double -> Int  -- A, lda
          -> IO ()

-- Single precision
foreign import ccall "cblas_sgemv"
    sgemv :: Int -> Int -> Int -> Int
          -> Float -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> Float -> Ptr Float -> Int
          -> IO ()

foreign import ccall "cblas_strmv"
    strmv :: Int -> Int -> Int -> Int
          -> Int -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> IO ()

foreign import ccall "cblas_strsv"
    strsv :: Int -> Int -> Int -> Int
          -> Int -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> IO ()

foreign import ccall "cblas_sger"
    sger :: Int -> Int -> Int
         -> Float -> Ptr Float -> Int
         -> Ptr Float -> Int
         -> Ptr Float -> Int
         -> IO ()

foreign import ccall "cblas_ssyr"
    ssyr :: Int -> Int -> Int
         -> Float -> Ptr Float -> Int
         -> Ptr Float -> Int
         -> IO ()

foreign import ccall "cblas_ssyr2"
    ssyr2 :: Int -> Int -> Int
          -> Float -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> IO ()

-- ============================================================
-- Level 3 BLAS (Matrix-Matrix)
-- ============================================================

-- | General matrix-matrix multiplication.
--
-- @C := alpha * op(A) * op(B) + beta * C@
--
-- ==== __Complexity__
--
-- O(M * N * K) operations.
foreign import ccall "cblas_dgemm"
    dgemm :: Int -> Int -> Int  -- Layout, TransA, TransB
          -> Int -> Int -> Int  -- M, N, K
          -> Double -> Ptr Double -> Int  -- alpha, A, lda
          -> Ptr Double -> Int  -- B, ldb
          -> Double -> Ptr Double -> Int  -- beta, C, ldc
          -> IO ()

-- | Symmetric matrix-matrix multiply.
foreign import ccall "cblas_dsymm"
    dsymm :: Int -> Int -> Int  -- Layout, Side, UpLo
          -> Int -> Int  -- M, N
          -> Double -> Ptr Double -> Int  -- alpha, A, lda
          -> Ptr Double -> Int  -- B, ldb
          -> Double -> Ptr Double -> Int  -- beta, C, ldc
          -> IO ()

-- | Symmetric rank-k update.
foreign import ccall "cblas_dsyrk"
    dsyrk :: Int -> Int -> Int  -- Layout, UpLo, Trans
          -> Int -> Int  -- N, K
          -> Double -> Ptr Double -> Int  -- alpha, A, lda
          -> Double -> Ptr Double -> Int  -- beta, C, ldc
          -> IO ()

-- | Symmetric rank-2k update.
foreign import ccall "cblas_dsyr2k"
    dsyr2k :: Int -> Int -> Int  -- Layout, UpLo, Trans
           -> Int -> Int  -- N, K
           -> Double -> Ptr Double -> Int  -- alpha, A, lda
           -> Ptr Double -> Int  -- B, ldb
           -> Double -> Ptr Double -> Int  -- beta, C, ldc
           -> IO ()

-- | Triangular matrix-matrix multiply.
foreign import ccall "cblas_dtrmm"
    dtrmm :: Int -> Int -> Int -> Int -> Int  -- Layout, Side, UpLo, Trans, Diag
          -> Int -> Int  -- M, N
          -> Double -> Ptr Double -> Int  -- alpha, A, lda
          -> Ptr Double -> Int  -- B, ldb
          -> IO ()

-- | Triangular solve with multiple right-hand sides.
foreign import ccall "cblas_dtrsm"
    dtrsm :: Int -> Int -> Int -> Int -> Int  -- Layout, Side, UpLo, Trans, Diag
          -> Int -> Int  -- M, N
          -> Double -> Ptr Double -> Int  -- alpha, A, lda
          -> Ptr Double -> Int  -- B, ldb
          -> IO ()

-- Single precision
foreign import ccall "cblas_sgemm"
    sgemm :: Int -> Int -> Int
          -> Int -> Int -> Int
          -> Float -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> Float -> Ptr Float -> Int
          -> IO ()

foreign import ccall "cblas_ssymm"
    ssymm :: Int -> Int -> Int
          -> Int -> Int
          -> Float -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> Float -> Ptr Float -> Int
          -> IO ()

foreign import ccall "cblas_ssyrk"
    ssyrk :: Int -> Int -> Int
          -> Int -> Int
          -> Float -> Ptr Float -> Int
          -> Float -> Ptr Float -> Int
          -> IO ()

foreign import ccall "cblas_ssyr2k"
    ssyr2k :: Int -> Int -> Int
           -> Int -> Int
           -> Float -> Ptr Float -> Int
           -> Ptr Float -> Int
           -> Float -> Ptr Float -> Int
           -> IO ()

foreign import ccall "cblas_strmm"
    strmm :: Int -> Int -> Int -> Int -> Int
          -> Int -> Int
          -> Float -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> IO ()

foreign import ccall "cblas_strsm"
    strsm :: Int -> Int -> Int -> Int -> Int
          -> Int -> Int
          -> Float -> Ptr Float -> Int
          -> Ptr Float -> Int
          -> IO ()

-- ============================================================
-- Internal FFI (provider checks)
-- ============================================================

foreign import ccall "bhc_openblas_available"
    c_openblas_available :: IO Bool

foreign import ccall "bhc_mkl_available"
    c_mkl_available :: IO Bool

foreign import ccall "bhc_accelerate_available"
    c_accelerate_available :: IO Bool

-- OpenBLAS/Accelerate CBLAS bindings (shared)
foreign import ccall "cblas_ddot"
    c_cblas_ddot :: Int -> Ptr Double -> Int -> Ptr Double -> Int -> IO Double

foreign import ccall "cblas_dnrm2"
    c_cblas_dnrm2 :: Int -> Ptr Double -> Int -> IO Double

foreign import ccall "cblas_daxpy"
    c_cblas_daxpy :: Int -> Double -> Ptr Double -> Int -> Ptr Double -> Int -> IO ()

foreign import ccall "cblas_dscal"
    c_cblas_dscal :: Int -> Double -> Ptr Double -> Int -> IO ()

foreign import ccall "cblas_dgemv"
    c_cblas_dgemv :: Layout -> Transpose -> Int -> Int
                  -> Double -> Ptr Double -> Int
                  -> Ptr Double -> Int
                  -> Double -> Ptr Double -> Int -> IO ()

foreign import ccall "cblas_dgemm"
    c_cblas_dgemm :: Layout -> Transpose -> Transpose
                  -> Int -> Int -> Int
                  -> Double -> Ptr Double -> Int
                  -> Ptr Double -> Int
                  -> Double -> Ptr Double -> Int -> IO ()

-- Intel MKL bindings
foreign import ccall "mkl_ddot"
    c_mkl_ddot :: Int -> Ptr Double -> Int -> Ptr Double -> Int -> IO Double

foreign import ccall "mkl_dnrm2"
    c_mkl_dnrm2 :: Int -> Ptr Double -> Int -> IO Double

foreign import ccall "mkl_daxpy"
    c_mkl_daxpy :: Int -> Double -> Ptr Double -> Int -> Ptr Double -> Int -> IO ()

foreign import ccall "mkl_dscal"
    c_mkl_dscal :: Int -> Double -> Ptr Double -> Int -> IO ()

foreign import ccall "mkl_dgemv"
    c_mkl_dgemv :: Layout -> Transpose -> Int -> Int
                -> Double -> Ptr Double -> Int
                -> Ptr Double -> Int
                -> Double -> Ptr Double -> Int -> IO ()

foreign import ccall "mkl_dgemm"
    c_mkl_dgemm :: Layout -> Transpose -> Transpose
                -> Int -> Int -> Int
                -> Double -> Ptr Double -> Int
                -> Ptr Double -> Int
                -> Double -> Ptr Double -> Int -> IO ()

-- Pure BHC fallback
foreign import ccall "bhc_pure_ddot"
    c_bhc_ddot :: Int -> Ptr Double -> Int -> Ptr Double -> Int -> IO Double

foreign import ccall "bhc_pure_dnrm2"
    c_bhc_dnrm2 :: Int -> Ptr Double -> Int -> IO Double

foreign import ccall "bhc_pure_daxpy"
    c_bhc_daxpy :: Int -> Double -> Ptr Double -> Int -> Ptr Double -> Int -> IO ()

foreign import ccall "bhc_pure_dscal"
    c_bhc_dscal :: Int -> Double -> Ptr Double -> Int -> IO ()

foreign import ccall "bhc_pure_dgemv"
    c_bhc_dgemv :: Layout -> Transpose -> Int -> Int
                -> Double -> Ptr Double -> Int
                -> Ptr Double -> Int
                -> Double -> Ptr Double -> Int -> IO ()

foreign import ccall "bhc_pure_dgemm"
    c_bhc_dgemm :: Layout -> Transpose -> Transpose
                -> Int -> Int -> Int
                -> Double -> Ptr Double -> Int
                -> Ptr Double -> Int
                -> Double -> Ptr Double -> Int -> IO ()
