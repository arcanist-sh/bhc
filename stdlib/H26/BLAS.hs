-- |
-- Module      : H26.BLAS
-- Description : Basic Linear Algebra Subprograms
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Linear algebra operations following the BLAS interface.
--
-- = Overview
--
-- This module provides the standard BLAS (Basic Linear Algebra Subprograms)
-- interface with pluggable backends. Supports:
--
-- * Level 1: Vector-vector operations (dot, axpy, nrm2)
-- * Level 2: Matrix-vector operations (gemv, trsv)
-- * Level 3: Matrix-matrix operations (gemm, trsm)
-- * High-level operations (matmul, solve, decompositions)
--
-- = Quick Start
--
-- @
-- {-# PROFILE Numeric #-}
-- import H26.BLAS
--
-- -- Matrix multiplication
-- c = matmul a b
-- c' = a @@ b              -- Infix version
--
-- -- GEMM: C = α·A·B + β·C
-- d = gemm NoTrans NoTrans 1.0 a b 0.0 c
--
-- -- Solve linear system A·X = B
-- x = solve a b
--
-- -- Decompositions
-- (l, u, p) = lu matrix
-- (q, r) = qr matrix
-- (u, s, vt) = svd matrix
-- @
--
-- = Backend Selection
--
-- BHC automatically selects the best available BLAS backend:
--
-- @
-- -- Check available backends
-- backends <- availableBackends
-- -- [MKL, OpenBLAS, PureHaskell]
--
-- -- Force specific backend
-- setBackend OpenBLAS
-- @
--
-- | Backend          | Description                         |
-- |------------------|-------------------------------------|
-- | MKL              | Intel Math Kernel Library           |
-- | OpenBLAS         | Open-source optimized BLAS          |
-- | AppleAccelerate  | Apple's Accelerate framework        |
-- | CUDA             | NVIDIA GPU via cuBLAS               |
-- | PureHaskell      | Fallback pure implementation        |
--
-- = Matrix Storage
--
-- Matrices can be stored in row-major (C) or column-major (Fortran) order.
-- Most operations accept both but may be faster with one or the other.
--
-- = See Also
--
-- * "H26.Tensor" for general tensor operations
-- * "H26.Numeric" for SIMD primitives
-- * "BHC.Numeric.BLAS" for the underlying implementation

{-# HASKELL_EDITION 2026 #-}
{-# PROFILE Numeric #-}

module H26.BLAS
  ( -- * BLAS Backend Configuration
    BLASBackend(..)
  , setBackend
  , getBackend
  , availableBackends

    -- * Matrix Types
  , Matrix
  , MatrixOrder(..)
  , Transpose(..)
  , UpLo(..)
  , Diag(..)
  , Side(..)

    -- * Matrix Construction
  , matrix
  , fromRows
  , fromCols
  , fromList
  , zeros
  , ones
  , identity
  , diagonal
  , fill

    -- * Matrix Properties
  , rows
  , cols
  , shape
  , size
  , order
  , isSquare
  , isSymmetric
  , isHermitian
  , isUpperTriangular
  , isLowerTriangular

    -- * Matrix Access
  , (!)
  , (!?)
  , row
  , col
  , diag
  , slice
  , submatrix

    -- * Matrix Manipulation
  , transpose
  , conjugateTranspose
  , reshape
  , flatten
  , asContiguous
  , asRowMajor
  , asColMajor

    -- * Level 1 BLAS (Vector Operations)
    -- ** Rotation
  , rotg      -- Generate Givens rotation
  , rotmg     -- Generate modified Givens rotation
  , rot       -- Apply Givens rotation
  , rotm      -- Apply modified Givens rotation

    -- ** Swap
  , swap      -- Swap vectors

    -- ** Scale
  , scal      -- Scale vector: x = α·x

    -- ** Copy
  , copy      -- Copy vector: y = x

    -- ** Axpy
  , axpy      -- Vector update: y = α·x + y

    -- ** Dot Products
  , dot       -- Dot product: x·y
  , dotu      -- Unconjugated dot product
  , dotc      -- Conjugated dot product
  , sdot      -- Single precision accumulation

    -- ** Norms
  , nrm2      -- Euclidean norm: ||x||₂
  , asum      -- Sum of absolute values: Σ|xᵢ|
  , iamax     -- Index of max absolute value

    -- * Level 2 BLAS (Matrix-Vector Operations)
    -- ** General Matrix-Vector
  , gemv      -- y = α·A·x + β·y
  , gbmv      -- Banded matrix-vector

    -- ** Hermitian/Symmetric Matrix-Vector
  , hemv      -- Hermitian matrix-vector
  , hbmv      -- Hermitian banded matrix-vector
  , hpmv      -- Hermitian packed matrix-vector
  , symv      -- Symmetric matrix-vector
  , sbmv      -- Symmetric banded matrix-vector
  , spmv      -- Symmetric packed matrix-vector

    -- ** Triangular Matrix-Vector
  , trmv      -- Triangular matrix-vector
  , tbmv      -- Triangular banded matrix-vector
  , tpmv      -- Triangular packed matrix-vector

    -- ** Triangular Solve
  , trsv      -- Triangular solve: x = A⁻¹·x
  , tbsv      -- Triangular banded solve
  , tpsv      -- Triangular packed solve

    -- ** Rank-1 Updates
  , ger       -- Rank-1 update: A = α·x·yᵀ + A
  , geru      -- Unconjugated rank-1 update
  , gerc      -- Conjugated rank-1 update
  , her       -- Hermitian rank-1 update
  , hpr       -- Hermitian packed rank-1 update
  , her2      -- Hermitian rank-2 update
  , hpr2      -- Hermitian packed rank-2 update
  , syr       -- Symmetric rank-1 update
  , spr       -- Symmetric packed rank-1 update
  , syr2      -- Symmetric rank-2 update
  , spr2      -- Symmetric packed rank-2 update

    -- * Level 3 BLAS (Matrix-Matrix Operations)
    -- ** General Matrix-Matrix
  , gemm      -- C = α·A·B + β·C
  , gemmBatched  -- Batched matrix multiply

    -- ** Symmetric/Hermitian Matrix-Matrix
  , symm      -- Symmetric matrix-matrix
  , hemm      -- Hermitian matrix-matrix
  , syrk      -- Symmetric rank-k update
  , herk      -- Hermitian rank-k update
  , syr2k     -- Symmetric rank-2k update
  , her2k     -- Hermitian rank-2k update

    -- ** Triangular Matrix-Matrix
  , trmm      -- Triangular matrix-matrix
  , trsm      -- Triangular matrix solve

    -- * High-Level Operations
    -- ** Matrix Multiply
  , matmul    -- Simple matrix multiply: C = A·B
  , (@@)      -- Infix matrix multiply

    -- ** Outer Product
  , outer     -- Outer product: A = x·yᵀ

    -- ** Vector Operations
  , vdot      -- Vector dot product
  , vnorm     -- Vector 2-norm
  , vnorm1    -- Vector 1-norm
  , vnormInf  -- Vector infinity norm
  , vscale    -- Scale vector
  , vadd      -- Vector addition
  , vsub      -- Vector subtraction
  , vmul      -- Element-wise multiply

    -- ** Matrix Operations
  , madd      -- Matrix addition
  , msub      -- Matrix subtraction
  , mmul      -- Element-wise multiply
  , mscale    -- Scale matrix
  , trace     -- Matrix trace
  , det       -- Determinant (via LU)
  , rank      -- Matrix rank
  , cond      -- Condition number

    -- ** Solvers
  , solve     -- Solve A·X = B
  , inv       -- Matrix inverse
  , pinv      -- Pseudo-inverse

    -- ** Decompositions (via LAPACK-style interface)
  , lu        -- LU decomposition
  , qr        -- QR decomposition
  , cholesky  -- Cholesky decomposition
  , svd       -- Singular value decomposition
  , eig       -- Eigenvalue decomposition
  , schur     -- Schur decomposition

    -- * Mutable Operations
  , MMatrix
  , thaw
  , freeze
  , unsafeThaw
  , unsafeFreeze
  , create

    -- * In-place Operations
  , gemmInPlace
  , scalInPlace
  , axpyInPlace
  , copyInPlace

    -- * Type Classes
  , BLASNum
  , BLASReal
  , BLASComplex
  ) where

-- | Available BLAS backends.
data BLASBackend
  = PureHaskell     -- ^ Pure Haskell implementation
  | OpenBLAS        -- ^ OpenBLAS library
  | MKL             -- ^ Intel Math Kernel Library
  | BLIS            -- ^ BLAS-like Library Instantiation Software
  | AppleAccelerate -- ^ Apple Accelerate framework
  | CUDA            -- ^ NVIDIA cuBLAS
  | ROCm            -- ^ AMD rocBLAS
  deriving (Eq, Show, Enum, Bounded)

-- | Set active BLAS backend.
setBackend :: BLASBackend -> IO ()

-- | Get current BLAS backend.
getBackend :: IO BLASBackend

-- | List available backends on this system.
availableBackends :: IO [BLASBackend]

-- | Dense matrix type.
--
-- Matrices are stored in either row-major or column-major order.
-- Shape is tracked at runtime for efficiency.
data Matrix a

-- | Mutable matrix for in-place operations.
data MMatrix s a

-- | Matrix storage order.
data MatrixOrder
  = RowMajor     -- ^ C-style row-major (default)
  | ColMajor     -- ^ Fortran-style column-major
  deriving (Eq, Show)

-- | Transpose operation specification.
data Transpose
  = NoTrans      -- ^ No transpose
  | Trans        -- ^ Transpose
  | ConjTrans    -- ^ Conjugate transpose
  deriving (Eq, Show)

-- | Upper/lower triangular specification.
data UpLo
  = Upper        -- ^ Upper triangular
  | Lower        -- ^ Lower triangular
  deriving (Eq, Show)

-- | Diagonal specification.
data Diag
  = NonUnit      -- ^ Non-unit diagonal
  | Unit         -- ^ Unit diagonal (implicit 1s)
  deriving (Eq, Show)

-- | Side specification for matrix operations.
data Side
  = LeftSide     -- ^ Operation from left
  | RightSide    -- ^ Operation from right
  deriving (Eq, Show)

-- Matrix Construction

-- | Create matrix from dimensions and function.
matrix :: BLASNum a => Int -> Int -> ((Int, Int) -> a) -> Matrix a

-- | Create matrix from list of rows.
fromRows :: BLASNum a => [[a]] -> Matrix a

-- | Create matrix from list of columns.
fromCols :: BLASNum a => [[a]] -> Matrix a

-- | Create matrix from flat list with dimensions.
fromList :: BLASNum a => Int -> Int -> [a] -> Matrix a

-- | Zero matrix.
zeros :: BLASNum a => Int -> Int -> Matrix a

-- | Matrix of ones.
ones :: BLASNum a => Int -> Int -> Matrix a

-- | Identity matrix.
identity :: BLASNum a => Int -> Matrix a

-- | Diagonal matrix from vector.
diagonal :: BLASNum a => [a] -> Matrix a

-- | Fill matrix with constant.
fill :: BLASNum a => Int -> Int -> a -> Matrix a

-- Matrix Properties

-- | Number of rows.
rows :: Matrix a -> Int

-- | Number of columns.
cols :: Matrix a -> Int

-- | Shape as (rows, cols).
shape :: Matrix a -> (Int, Int)

-- | Total number of elements.
size :: Matrix a -> Int

-- | Storage order.
order :: Matrix a -> MatrixOrder

-- | Check if square.
isSquare :: Matrix a -> Bool

-- | Check if symmetric.
isSymmetric :: BLASNum a => Matrix a -> Bool

-- | Check if Hermitian.
isHermitian :: BLASComplex a => Matrix a -> Bool

-- | Check if upper triangular.
isUpperTriangular :: BLASNum a => Matrix a -> Bool

-- | Check if lower triangular.
isLowerTriangular :: BLASNum a => Matrix a -> Bool

-- Matrix Access

-- | Index element (row, col).
(!) :: BLASNum a => Matrix a -> (Int, Int) -> a

-- | Safe indexing.
(!?) :: BLASNum a => Matrix a -> (Int, Int) -> Maybe a

-- | Extract row as vector.
row :: BLASNum a => Int -> Matrix a -> [a]

-- | Extract column as vector.
col :: BLASNum a => Int -> Matrix a -> [a]

-- | Extract diagonal.
diag :: BLASNum a => Matrix a -> [a]

-- | Slice submatrix.
slice :: BLASNum a => (Int, Int) -> (Int, Int) -> Matrix a -> Matrix a

-- | Extract submatrix.
submatrix :: BLASNum a => Int -> Int -> Int -> Int -> Matrix a -> Matrix a

-- Matrix Manipulation

-- | Transpose matrix.
transpose :: BLASNum a => Matrix a -> Matrix a

-- | Conjugate transpose.
conjugateTranspose :: BLASComplex a => Matrix a -> Matrix a

-- | Reshape matrix.
reshape :: BLASNum a => Int -> Int -> Matrix a -> Matrix a

-- | Flatten to vector.
flatten :: BLASNum a => Matrix a -> [a]

-- | Force contiguous storage.
asContiguous :: BLASNum a => Matrix a -> Matrix a

-- | Convert to row-major order.
asRowMajor :: BLASNum a => Matrix a -> Matrix a

-- | Convert to column-major order.
asColMajor :: BLASNum a => Matrix a -> Matrix a

-- Level 1 BLAS

-- | Generate Givens rotation.
rotg :: BLASReal a => a -> a -> (a, a, a, a)

-- | Generate modified Givens rotation.
rotmg :: BLASReal a => a -> a -> a -> a -> (a, [a])

-- | Apply Givens rotation.
rot :: BLASReal a => [a] -> [a] -> a -> a -> ([a], [a])

-- | Apply modified Givens rotation.
rotm :: BLASReal a => [a] -> [a] -> [a] -> ([a], [a])

-- | Swap two vectors.
swap :: BLASNum a => [a] -> [a] -> ([a], [a])

-- | Scale vector: x = α·x
scal :: BLASNum a => a -> [a] -> [a]

-- | Copy vector: y = x
copy :: BLASNum a => [a] -> [a]

-- | Vector update: y = α·x + y
axpy :: BLASNum a => a -> [a] -> [a] -> [a]

-- | Dot product: x·y
dot :: BLASReal a => [a] -> [a] -> a

-- | Unconjugated complex dot product.
dotu :: BLASComplex a => [a] -> [a] -> a

-- | Conjugated complex dot product.
dotc :: BLASComplex a => [a] -> [a] -> a

-- | Single precision dot with double accumulation.
sdot :: [Float] -> [Float] -> Double

-- | Euclidean norm: ||x||₂
nrm2 :: BLASNum a => [a] -> a

-- | Sum of absolute values.
asum :: BLASNum a => [a] -> a

-- | Index of maximum absolute value.
iamax :: BLASNum a => [a] -> Int

-- Level 2 BLAS

-- | General matrix-vector: y = α·op(A)·x + β·y
gemv :: BLASNum a
     => Transpose -> a -> Matrix a -> [a] -> a -> [a] -> [a]

-- | General banded matrix-vector.
gbmv :: BLASNum a
     => Transpose -> Int -> Int -> a -> Matrix a -> [a] -> a -> [a] -> [a]

-- | Hermitian matrix-vector: y = α·A·x + β·y
hemv :: BLASComplex a
     => UpLo -> a -> Matrix a -> [a] -> a -> [a] -> [a]

-- | Hermitian banded matrix-vector.
hbmv :: BLASComplex a
     => UpLo -> Int -> a -> Matrix a -> [a] -> a -> [a] -> [a]

-- | Hermitian packed matrix-vector.
hpmv :: BLASComplex a
     => UpLo -> a -> [a] -> [a] -> a -> [a] -> [a]

-- | Symmetric matrix-vector: y = α·A·x + β·y
symv :: BLASReal a
     => UpLo -> a -> Matrix a -> [a] -> a -> [a] -> [a]

-- | Symmetric banded matrix-vector.
sbmv :: BLASReal a
     => UpLo -> Int -> a -> Matrix a -> [a] -> a -> [a] -> [a]

-- | Symmetric packed matrix-vector.
spmv :: BLASReal a
     => UpLo -> a -> [a] -> [a] -> a -> [a] -> [a]

-- | Triangular matrix-vector: x = op(A)·x
trmv :: BLASNum a
     => UpLo -> Transpose -> Diag -> Matrix a -> [a] -> [a]

-- | Triangular banded matrix-vector.
tbmv :: BLASNum a
     => UpLo -> Transpose -> Diag -> Int -> Matrix a -> [a] -> [a]

-- | Triangular packed matrix-vector.
tpmv :: BLASNum a
     => UpLo -> Transpose -> Diag -> [a] -> [a] -> [a]

-- | Triangular solve: x = op(A)⁻¹·x
trsv :: BLASNum a
     => UpLo -> Transpose -> Diag -> Matrix a -> [a] -> [a]

-- | Triangular banded solve.
tbsv :: BLASNum a
     => UpLo -> Transpose -> Diag -> Int -> Matrix a -> [a] -> [a]

-- | Triangular packed solve.
tpsv :: BLASNum a
     => UpLo -> Transpose -> Diag -> [a] -> [a] -> [a]

-- | Rank-1 update: A = α·x·yᵀ + A
ger :: BLASReal a => a -> [a] -> [a] -> Matrix a -> Matrix a

-- | Unconjugated rank-1 update.
geru :: BLASComplex a => a -> [a] -> [a] -> Matrix a -> Matrix a

-- | Conjugated rank-1 update.
gerc :: BLASComplex a => a -> [a] -> [a] -> Matrix a -> Matrix a

-- | Hermitian rank-1 update: A = α·x·x^H + A
her :: BLASComplex a => UpLo -> a -> [a] -> Matrix a -> Matrix a

-- | Hermitian packed rank-1 update.
hpr :: BLASComplex a => UpLo -> a -> [a] -> [a] -> [a]

-- | Hermitian rank-2 update.
her2 :: BLASComplex a => UpLo -> a -> [a] -> [a] -> Matrix a -> Matrix a

-- | Hermitian packed rank-2 update.
hpr2 :: BLASComplex a => UpLo -> a -> [a] -> [a] -> [a] -> [a]

-- | Symmetric rank-1 update: A = α·x·xᵀ + A
syr :: BLASReal a => UpLo -> a -> [a] -> Matrix a -> Matrix a

-- | Symmetric packed rank-1 update.
spr :: BLASReal a => UpLo -> a -> [a] -> [a] -> [a]

-- | Symmetric rank-2 update.
syr2 :: BLASReal a => UpLo -> a -> [a] -> [a] -> Matrix a -> Matrix a

-- | Symmetric packed rank-2 update.
spr2 :: BLASReal a => UpLo -> a -> [a] -> [a] -> [a] -> [a]

-- Level 3 BLAS

-- | General matrix-matrix: C = α·op(A)·op(B) + β·C
gemm :: BLASNum a
     => Transpose -> Transpose
     -> a -> Matrix a -> Matrix a -> a -> Matrix a -> Matrix a

-- | Batched matrix multiply.
gemmBatched :: BLASNum a
            => Transpose -> Transpose
            -> a -> [Matrix a] -> [Matrix a] -> a -> [Matrix a] -> [Matrix a]

-- | Symmetric matrix-matrix: C = α·A·B + β·C or C = α·B·A + β·C
symm :: BLASReal a
     => Side -> UpLo -> a -> Matrix a -> Matrix a -> a -> Matrix a -> Matrix a

-- | Hermitian matrix-matrix.
hemm :: BLASComplex a
     => Side -> UpLo -> a -> Matrix a -> Matrix a -> a -> Matrix a -> Matrix a

-- | Symmetric rank-k update: C = α·A·Aᵀ + β·C
syrk :: BLASReal a
     => UpLo -> Transpose -> a -> Matrix a -> a -> Matrix a -> Matrix a

-- | Hermitian rank-k update: C = α·A·A^H + β·C
herk :: BLASComplex a
     => UpLo -> Transpose -> a -> Matrix a -> a -> Matrix a -> Matrix a

-- | Symmetric rank-2k update.
syr2k :: BLASReal a
      => UpLo -> Transpose -> a -> Matrix a -> Matrix a -> a -> Matrix a -> Matrix a

-- | Hermitian rank-2k update.
her2k :: BLASComplex a
      => UpLo -> Transpose -> a -> Matrix a -> Matrix a -> a -> Matrix a -> Matrix a

-- | Triangular matrix-matrix: B = α·op(A)·B or B = α·B·op(A)
trmm :: BLASNum a
     => Side -> UpLo -> Transpose -> Diag -> a -> Matrix a -> Matrix a -> Matrix a

-- | Triangular solve: B = α·op(A)⁻¹·B or B = α·B·op(A)⁻¹
trsm :: BLASNum a
     => Side -> UpLo -> Transpose -> Diag -> a -> Matrix a -> Matrix a -> Matrix a

-- High-Level Operations

-- | Simple matrix multiply: C = A·B
matmul :: BLASNum a => Matrix a -> Matrix a -> Matrix a

-- | Infix matrix multiply.
(@@) :: BLASNum a => Matrix a -> Matrix a -> Matrix a
infixl 7 @@

-- | Outer product: A = x·yᵀ
outer :: BLASNum a => [a] -> [a] -> Matrix a

-- | Vector dot product.
vdot :: BLASNum a => [a] -> [a] -> a

-- | Vector 2-norm.
vnorm :: BLASNum a => [a] -> a

-- | Vector 1-norm.
vnorm1 :: BLASNum a => [a] -> a

-- | Vector infinity norm.
vnormInf :: BLASNum a => [a] -> a

-- | Scale vector.
vscale :: BLASNum a => a -> [a] -> [a]

-- | Vector addition.
vadd :: BLASNum a => [a] -> [a] -> [a]

-- | Vector subtraction.
vsub :: BLASNum a => [a] -> [a] -> [a]

-- | Element-wise vector multiply.
vmul :: BLASNum a => [a] -> [a] -> [a]

-- | Matrix addition.
madd :: BLASNum a => Matrix a -> Matrix a -> Matrix a

-- | Matrix subtraction.
msub :: BLASNum a => Matrix a -> Matrix a -> Matrix a

-- | Element-wise matrix multiply.
mmul :: BLASNum a => Matrix a -> Matrix a -> Matrix a

-- | Scale matrix.
mscale :: BLASNum a => a -> Matrix a -> Matrix a

-- | Matrix trace.
trace :: BLASNum a => Matrix a -> a

-- | Matrix determinant (via LU decomposition).
det :: BLASNum a => Matrix a -> a

-- | Matrix rank.
rank :: BLASNum a => Matrix a -> Int

-- | Condition number.
cond :: BLASNum a => Matrix a -> a

-- Solvers

-- | Solve linear system A·X = B.
solve :: BLASNum a => Matrix a -> Matrix a -> Matrix a

-- | Matrix inverse.
inv :: BLASNum a => Matrix a -> Matrix a

-- | Moore-Penrose pseudo-inverse.
pinv :: BLASNum a => Matrix a -> Matrix a

-- Decompositions

-- | LU decomposition with partial pivoting.
--
-- Returns (L, U, P) such that P·A = L·U.
lu :: BLASNum a => Matrix a -> (Matrix a, Matrix a, [Int])

-- | QR decomposition.
--
-- Returns (Q, R) such that A = Q·R.
qr :: BLASNum a => Matrix a -> (Matrix a, Matrix a)

-- | Cholesky decomposition.
--
-- Returns L such that A = L·Lᵀ (or A = L·L^H for complex).
-- Matrix must be positive definite.
cholesky :: BLASNum a => UpLo -> Matrix a -> Matrix a

-- | Singular value decomposition.
--
-- Returns (U, S, Vt) such that A = U·diag(S)·Vt.
svd :: BLASNum a => Matrix a -> (Matrix a, [a], Matrix a)

-- | Eigenvalue decomposition.
--
-- Returns (eigenvalues, eigenvectors).
eig :: BLASNum a => Matrix a -> ([a], Matrix a)

-- | Schur decomposition.
--
-- Returns (T, Z) such that A = Z·T·Zᵀ.
schur :: BLASNum a => Matrix a -> (Matrix a, Matrix a)

-- Mutable Operations

-- | Convert to mutable matrix.
thaw :: BLASNum a => Matrix a -> ST s (MMatrix s a)

-- | Convert to immutable matrix.
freeze :: BLASNum a => MMatrix s a -> ST s (Matrix a)

-- | Unsafe thaw (no copy).
unsafeThaw :: BLASNum a => Matrix a -> ST s (MMatrix s a)

-- | Unsafe freeze (no copy).
unsafeFreeze :: BLASNum a => MMatrix s a -> ST s (Matrix a)

-- | Create matrix via mutable computation.
create :: BLASNum a => (forall s. ST s (MMatrix s a)) -> Matrix a

-- In-place Operations

-- | In-place GEMM.
gemmInPlace :: BLASNum a
            => Transpose -> Transpose
            -> a -> Matrix a -> Matrix a -> a -> MMatrix s a -> ST s ()

-- | In-place scale.
scalInPlace :: BLASNum a => a -> MMatrix s a -> ST s ()

-- | In-place axpy.
axpyInPlace :: BLASNum a => a -> [a] -> MMatrix s a -> ST s ()

-- | In-place copy.
copyInPlace :: BLASNum a => Matrix a -> MMatrix s a -> ST s ()

-- Type Classes

-- | Types supporting BLAS operations.
class (Eq a, Num a) => BLASNum a where
  blasZero :: a
  blasOne :: a

-- | Real BLAS types (Float, Double).
class BLASNum a => BLASReal a

-- | Complex BLAS types.
class BLASNum a => BLASComplex a where
  conjg :: a -> a
  realPart :: a -> a
  imagPart :: a -> a

-- Instances
instance BLASNum Float
instance BLASNum Double
instance BLASReal Float
instance BLASReal Double

-- Internal types
data ST s a

-- This is a specification file.
-- Actual implementation provided by the compiler.
