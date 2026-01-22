-- |
-- Module      : BHC.Numeric.Matrix
-- Description : Dense matrix operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Dense 2D matrices with BLAS-accelerated operations.
-- Matrices are stored in row-major order.

{-# LANGUAGE BangPatterns #-}

module BHC.Numeric.Matrix (
    -- * Matrix type
    Matrix,

    -- * Construction
    zeros, ones, full,
    fromLists, fromList,
    fromRows, fromCols,
    identity, eye,
    diag, diagFrom,

    -- * Properties
    rows, cols, size,
    shape,

    -- * Indexing
    (!), (!?),
    row, col,
    getRow, getCol,
    getDiag,

    -- * Slicing
    submatrix,
    takeRows, dropRows,
    takeCols, dropCols,

    -- * Combining
    (|||), (---),
    hcat, vcat,
    fromBlocks,

    -- * Element-wise operations
    map, imap,
    zipWith, izipWith,

    -- * Matrix operations
    transpose, (^.),
    (+.), (-.), (*.), (/.),
    scale,

    -- * Matrix multiplication
    (@@), mul, mulV,

    -- * Linear algebra
    trace, det, rank,
    inv, pinv,
    solve, lstsq,

    -- * Decompositions
    lu, qr, svd,
    cholesky,
    eig, eigvals,

    -- * Norms
    norm1, norm2, normInf, normFrob,

    -- * Folds
    sum, product,
    maximum, minimum,
    sumRows, sumCols,
    meanRows, meanCols,

    -- * Conversion
    toLists, toList,
    flatten,
    asVector, asColumn,
) where

import BHC.Prelude hiding (
    map, zipWith, sum, product, maximum, minimum
    )
import qualified BHC.Prelude as P
import qualified BHC.Numeric.Vector as V

-- ============================================================
-- Matrix Type
-- ============================================================

-- | A dense 2D matrix.
data Matrix a = Matrix
    { matData    :: !MatrixData
    , matRows    :: !Int
    , matCols    :: !Int
    , matStride  :: !Int  -- Row stride
    , matOffset  :: !Int
    }

-- | Internal matrix storage.
data MatrixData

-- ============================================================
-- Construction
-- ============================================================

-- | Create matrix of zeros.
foreign import ccall "bhc_matrix_zeros"
    zeros :: Int -> Int -> IO (Matrix Float)

-- | Create matrix of ones.
foreign import ccall "bhc_matrix_ones"
    ones :: Int -> Int -> IO (Matrix Float)

-- | Create matrix filled with value.
full :: Int -> Int -> a -> Matrix a
full r c x = undefined

-- | Create matrix from nested lists.
--
-- >>> fromLists [[1, 2], [3, 4]]
-- Matrix 2x2 [[1, 2], [3, 4]]
fromLists :: [[a]] -> Matrix a
fromLists = undefined

-- | Create matrix from flat list with dimensions.
fromList :: Int -> Int -> [a] -> Matrix a
fromList r c xs = undefined

-- | Create matrix from row vectors.
fromRows :: [V.Vector a] -> Matrix a
fromRows = undefined

-- | Create matrix from column vectors.
fromCols :: [V.Vector a] -> Matrix a
fromCols = undefined

-- | Identity matrix.
--
-- >>> identity 3
-- Matrix 3x3 [[1,0,0], [0,1,0], [0,0,1]]
identity :: Num a => Int -> Matrix a
identity n = diag (P.replicate n 1)

-- | Identity matrix (alias for identity).
eye :: Num a => Int -> Matrix a
eye = identity

-- | Diagonal matrix from list.
diag :: [a] -> Matrix a
diag xs = undefined

-- | Diagonal matrix from vector.
diagFrom :: V.Vector a -> Matrix a
diagFrom = undefined

-- ============================================================
-- Properties
-- ============================================================

-- | Number of rows.
rows :: Matrix a -> Int
rows = matRows

-- | Number of columns.
cols :: Matrix a -> Int
cols = matCols

-- | Total number of elements.
size :: Matrix a -> Int
size m = rows m * cols m

-- | Shape as (rows, cols).
shape :: Matrix a -> (Int, Int)
shape m = (rows m, cols m)

-- ============================================================
-- Indexing
-- ============================================================

-- | Index into matrix (unsafe).
(!) :: Matrix a -> (Int, Int) -> a
(!) = undefined

-- | Index into matrix (safe).
(!?) :: Matrix a -> (Int, Int) -> Maybe a
m !? (i, j)
    | i < 0 || i >= rows m = Nothing
    | j < 0 || j >= cols m = Nothing
    | otherwise = Just (m ! (i, j))

-- | Extract single row as vector.
row :: Int -> Matrix a -> V.Vector a
row i m = undefined

-- | Extract single column as vector.
col :: Int -> Matrix a -> V.Vector a
col j m = undefined

-- | Get row (alias for row).
getRow :: Int -> Matrix a -> V.Vector a
getRow = row

-- | Get column (alias for col).
getCol :: Int -> Matrix a -> V.Vector a
getCol = col

-- | Get main diagonal.
getDiag :: Matrix a -> V.Vector a
getDiag m = undefined

-- ============================================================
-- Slicing
-- ============================================================

-- | Extract submatrix.
--
-- >>> submatrix 0 2 0 2 m  -- 2x2 top-left corner
submatrix :: Int -> Int -> Int -> Int -> Matrix a -> Matrix a
submatrix r1 r2 c1 c2 m = undefined

-- | Take first n rows.
takeRows :: Int -> Matrix a -> Matrix a
takeRows n m = submatrix 0 n 0 (cols m) m

-- | Drop first n rows.
dropRows :: Int -> Matrix a -> Matrix a
dropRows n m = submatrix n (rows m) 0 (cols m) m

-- | Take first n columns.
takeCols :: Int -> Matrix a -> Matrix a
takeCols n m = submatrix 0 (rows m) 0 n m

-- | Drop first n columns.
dropCols :: Int -> Matrix a -> Matrix a
dropCols n m = submatrix 0 (rows m) n (cols m) m

-- ============================================================
-- Combining
-- ============================================================

-- | Horizontal concatenation.
--
-- >>> a ||| b
-- [a b]
(|||) :: Matrix a -> Matrix a -> Matrix a
(|||) = undefined
infixr 5 |||

-- | Vertical concatenation.
--
-- >>> a --- b
-- [a]
-- [b]
(---) :: Matrix a -> Matrix a -> Matrix a
(---) = undefined
infixr 4 ---

-- | Horizontal concatenation of multiple matrices.
hcat :: [Matrix a] -> Matrix a
hcat = P.foldl1 (|||)

-- | Vertical concatenation of multiple matrices.
vcat :: [Matrix a] -> Matrix a
vcat = P.foldl1 (---)

-- | Create matrix from blocks.
--
-- >>> fromBlocks [[a, b], [c, d]]
fromBlocks :: [[Matrix a]] -> Matrix a
fromBlocks blocks = vcat (P.map hcat blocks)

-- ============================================================
-- Element-wise Operations
-- ============================================================

-- | Map function over elements.
map :: (a -> b) -> Matrix a -> Matrix b
map f m = undefined

-- | Map with indices.
imap :: (Int -> Int -> a -> b) -> Matrix a -> Matrix b
imap f m = undefined

-- | Zip two matrices.
zipWith :: (a -> b -> c) -> Matrix a -> Matrix b -> Matrix c
zipWith f ma mb = undefined

-- | Zip with indices.
izipWith :: (Int -> Int -> a -> b -> c) -> Matrix a -> Matrix b -> Matrix c
izipWith f ma mb = undefined

-- ============================================================
-- Matrix Operations
-- ============================================================

-- | Transpose matrix.
--
-- >>> transpose m
-- m^T
foreign import ccall "bhc_matrix_transpose"
    transpose :: Matrix a -> Matrix a

-- | Transpose operator.
(^.) :: Matrix a -> Matrix a
(^.) = transpose
infixl 8 ^.

-- | Element-wise addition.
(+.) :: Num a => Matrix a -> Matrix a -> Matrix a
(+.) = zipWith (+)
infixl 6 +.

-- | Element-wise subtraction.
(-.) :: Num a => Matrix a -> Matrix a -> Matrix a
(-.) = zipWith (-)
infixl 6 -.

-- | Element-wise multiplication (Hadamard product).
(*.) :: Num a => Matrix a -> Matrix a -> Matrix a
(*.) = zipWith (*)
infixl 7 *.

-- | Element-wise division.
(/.) :: Fractional a => Matrix a -> Matrix a -> Matrix a
(/.) = zipWith (/)
infixl 7 /.

-- | Scale matrix by scalar.
scale :: Num a => a -> Matrix a -> Matrix a
scale k = map (* k)

-- ============================================================
-- Matrix Multiplication
-- ============================================================

-- | Matrix multiplication.
--
-- ==== __Complexity__
--
-- O(n * m * k) for (n x m) @@ (m x k)
--
-- Uses BLAS DGEMM when available.
foreign import ccall "bhc_matrix_mul"
    mul :: Num a => Matrix a -> Matrix a -> Matrix a

-- | Matrix multiplication operator.
(@@) :: Num a => Matrix a -> Matrix a -> Matrix a
(@@) = mul
infixl 7 @@

-- | Matrix-vector multiplication.
--
-- >>> mulV m v  -- m @ v
foreign import ccall "bhc_matrix_vec_mul"
    mulV :: Num a => Matrix a -> V.Vector a -> V.Vector a

-- ============================================================
-- Linear Algebra
-- ============================================================

-- | Matrix trace (sum of diagonal).
trace :: Num a => Matrix a -> a
trace m = V.sum (getDiag m)

-- | Matrix determinant.
foreign import ccall "bhc_matrix_det"
    det :: Floating a => Matrix a -> a

-- | Matrix rank.
foreign import ccall "bhc_matrix_rank"
    rank :: Floating a => Matrix a -> Int

-- | Matrix inverse.
--
-- Throws error if matrix is singular.
foreign import ccall "bhc_matrix_inv"
    inv :: Floating a => Matrix a -> Matrix a

-- | Moore-Penrose pseudoinverse.
foreign import ccall "bhc_matrix_pinv"
    pinv :: Floating a => Matrix a -> Matrix a

-- | Solve linear system Ax = b.
--
-- Returns x such that Ax = b.
foreign import ccall "bhc_matrix_solve"
    solve :: Floating a => Matrix a -> V.Vector a -> V.Vector a

-- | Least squares solution.
--
-- Minimizes ||Ax - b||_2.
foreign import ccall "bhc_matrix_lstsq"
    lstsq :: Floating a => Matrix a -> V.Vector a -> V.Vector a

-- ============================================================
-- Decompositions
-- ============================================================

-- | LU decomposition.
--
-- Returns (L, U, P) where PA = LU.
foreign import ccall "bhc_matrix_lu"
    lu :: Floating a => Matrix a -> (Matrix a, Matrix a, Matrix a)

-- | QR decomposition.
--
-- Returns (Q, R) where A = QR.
foreign import ccall "bhc_matrix_qr"
    qr :: Floating a => Matrix a -> (Matrix a, Matrix a)

-- | Singular value decomposition.
--
-- Returns (U, S, V) where A = U * diag(S) * V^T.
foreign import ccall "bhc_matrix_svd"
    svd :: Floating a => Matrix a -> (Matrix a, V.Vector a, Matrix a)

-- | Cholesky decomposition.
--
-- Returns L where A = LL^T.
-- Requires A to be positive definite.
foreign import ccall "bhc_matrix_cholesky"
    cholesky :: Floating a => Matrix a -> Matrix a

-- | Eigendecomposition.
--
-- Returns (eigenvalues, eigenvectors).
-- Eigenvectors are columns of the matrix.
foreign import ccall "bhc_matrix_eig"
    eig :: Floating a => Matrix a -> (V.Vector (Complex a), Matrix (Complex a))

-- | Eigenvalues only.
foreign import ccall "bhc_matrix_eigvals"
    eigvals :: Floating a => Matrix a -> V.Vector (Complex a)

-- Complex number placeholder
data Complex a = Complex !a !a
    deriving (Eq, Show)

-- ============================================================
-- Norms
-- ============================================================

-- | 1-norm (maximum column sum).
norm1 :: Num a => Matrix a -> a
norm1 m = P.maximum [V.sum (V.map P.abs (col j m)) | j <- [0..cols m - 1]]

-- | 2-norm (spectral norm, largest singular value).
foreign import ccall "bhc_matrix_norm2"
    norm2 :: Floating a => Matrix a -> a

-- | Infinity norm (maximum row sum).
normInf :: Num a => Matrix a -> a
normInf m = P.maximum [V.sum (V.map P.abs (row i m)) | i <- [0..rows m - 1]]

-- | Frobenius norm (sqrt of sum of squares).
normFrob :: Floating a => Matrix a -> a
normFrob m = P.sqrt (P.sum [m ! (i, j) ^ 2 | i <- [0..rows m - 1], j <- [0..cols m - 1]])

-- ============================================================
-- Folds
-- ============================================================

-- | Sum of all elements.
sum :: Num a => Matrix a -> a
sum m = P.sum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Product of all elements.
product :: Num a => Matrix a -> a
product m = P.product [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Maximum element.
maximum :: Ord a => Matrix a -> a
maximum m = P.maximum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Minimum element.
minimum :: Ord a => Matrix a -> a
minimum m = P.minimum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Sum each row.
sumRows :: Num a => Matrix a -> V.Vector a
sumRows m = V.generate (rows m) (\i -> V.sum (row i m))

-- | Sum each column.
sumCols :: Num a => Matrix a -> V.Vector a
sumCols m = V.generate (cols m) (\j -> V.sum (col j m))

-- | Mean of each row.
meanRows :: Fractional a => Matrix a -> V.Vector a
meanRows m = V.map (/ P.fromIntegral (cols m)) (sumRows m)

-- | Mean of each column.
meanCols :: Fractional a => Matrix a -> V.Vector a
meanCols m = V.map (/ P.fromIntegral (rows m)) (sumCols m)

-- ============================================================
-- Conversion
-- ============================================================

-- | Convert to nested lists.
toLists :: Matrix a -> [[a]]
toLists m = [[m ! (i, j) | j <- [0..cols m - 1]] | i <- [0..rows m - 1]]

-- | Convert to flat list (row-major).
toList :: Matrix a -> [a]
toList = P.concat . toLists

-- | Flatten to vector.
flatten :: Matrix a -> V.Vector a
flatten m = V.fromList (toList m)

-- | View vector as 1-column matrix.
asColumn :: V.Vector a -> Matrix a
asColumn v = fromList (V.length v) 1 (V.toList v)

-- | View vector as 1-row matrix.
asVector :: V.Vector a -> Matrix a
asVector v = fromList 1 (V.length v) (V.toList v)
