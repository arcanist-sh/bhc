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
{-# LANGUAGE ForeignFunctionInterface #-}

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

    -- * Type class
    MatrixElem,
) where

import BHC.Prelude hiding (
    map, zipWith, sum, product, maximum, minimum
    )
import qualified BHC.Prelude as P
import qualified BHC.Numeric.Vector as V
import Foreign.Ptr (Ptr, nullPtr, FunPtr)
import Foreign.ForeignPtr (ForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Array (withArrayLen)
import System.IO.Unsafe (unsafePerformIO)

-- ============================================================
-- FFI Imports for f64 matrices
-- ============================================================

foreign import ccall unsafe "bhc_matrix_from_f64"
    c_matrix_from_f64 :: Ptr Double -> Int -> Int -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_free_f64"
    c_matrix_free_f64 :: Ptr MatrixData -> IO ()

foreign import ccall unsafe "&bhc_matrix_free_f64"
    c_matrix_finalizer_f64 :: FunPtr (Ptr MatrixData -> IO ())

foreign import ccall unsafe "bhc_matrix_rows_f64"
    c_matrix_rows_f64 :: Ptr MatrixData -> IO Int

foreign import ccall unsafe "bhc_matrix_cols_f64"
    c_matrix_cols_f64 :: Ptr MatrixData -> IO Int

foreign import ccall unsafe "bhc_matrix_get_f64"
    c_matrix_get_f64 :: Ptr MatrixData -> Int -> Int -> IO Double

foreign import ccall unsafe "bhc_matrix_zeros_f64"
    c_matrix_zeros_f64 :: Int -> Int -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_identity_f64"
    c_matrix_identity_f64 :: Int -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_matmul_f64"
    c_matrix_matmul_f64 :: Ptr MatrixData -> Ptr MatrixData -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_transpose_f64"
    c_matrix_transpose_f64 :: Ptr MatrixData -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_add_f64"
    c_matrix_add_f64 :: Ptr MatrixData -> Ptr MatrixData -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_scale_f64"
    c_matrix_scale_f64 :: Ptr MatrixData -> Double -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_trace_f64"
    c_matrix_trace_f64 :: Ptr MatrixData -> IO Double

foreign import ccall unsafe "bhc_matrix_norm_f64"
    c_matrix_norm_f64 :: Ptr MatrixData -> IO Double

-- ============================================================
-- Matrix Type
-- ============================================================

-- | A dense 2D matrix using foreign memory.
data Matrix a = Matrix
    { matPtr     :: !(ForeignPtr MatrixData)
    , matRows    :: !Int
    , matCols    :: !Int
    , matStride  :: !Int  -- Row stride
    , matOffset  :: !Int
    }

-- | Internal matrix storage (opaque Rust type).
data MatrixData

-- | Type class for matrix element operations
class MatrixElem a where
    matrixFromList :: Int -> Int -> [a] -> IO (Matrix a)
    matrixGet :: Matrix a -> Int -> Int -> IO a
    matrixZeros :: Int -> Int -> IO (Matrix a)
    matrixIdentity :: Int -> IO (Matrix a)
    matrixMatmul :: Matrix a -> Matrix a -> IO (Matrix a)
    matrixTranspose :: Matrix a -> IO (Matrix a)
    matrixAdd :: Matrix a -> Matrix a -> IO (Matrix a)
    matrixScale :: a -> Matrix a -> IO (Matrix a)
    matrixTrace :: Matrix a -> IO a
    matrixNorm :: Matrix a -> IO a

instance MatrixElem Double where
    matrixFromList r c xs = do
        withArrayLen xs $ \len ptr -> do
            if len /= r * c
                then error "Matrix dimensions don't match data length"
                else do
                    mptr <- c_matrix_from_f64 ptr r c
                    if mptr == nullPtr
                        then error "Failed to create matrix"
                        else do
                            fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                            return $ Matrix fp r c c 0
    matrixGet (Matrix fp _ _ _ _) row col = withForeignPtr fp $ \ptr ->
        c_matrix_get_f64 ptr row col
    matrixZeros r c = do
        mptr <- c_matrix_zeros_f64 r c
        if mptr == nullPtr
            then error "Failed to create zeros matrix"
            else do
                fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                return $ Matrix fp r c c 0
    matrixIdentity n = do
        mptr <- c_matrix_identity_f64 n
        if mptr == nullPtr
            then error "Failed to create identity matrix"
            else do
                fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                return $ Matrix fp n n n 0
    matrixMatmul (Matrix fp1 r1 c1 _ _) (Matrix fp2 r2 c2 _ _) =
        withForeignPtr fp1 $ \p1 ->
        withForeignPtr fp2 $ \p2 -> do
            mptr <- c_matrix_matmul_f64 p1 p2
            if mptr == nullPtr
                then error "Matrix multiplication failed (dimension mismatch)"
                else do
                    fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                    return $ Matrix fp r1 c2 c2 0
    matrixTranspose (Matrix fp r c _ _) =
        withForeignPtr fp $ \ptr -> do
            mptr <- c_matrix_transpose_f64 ptr
            if mptr == nullPtr
                then error "Matrix transpose failed"
                else do
                    fp' <- newForeignPtr c_matrix_finalizer_f64 mptr
                    return $ Matrix fp' c r r 0
    matrixAdd (Matrix fp1 r c _ _) (Matrix fp2 _ _ _ _) =
        withForeignPtr fp1 $ \p1 ->
        withForeignPtr fp2 $ \p2 -> do
            mptr <- c_matrix_add_f64 p1 p2
            if mptr == nullPtr
                then error "Matrix addition failed (dimension mismatch)"
                else do
                    fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                    return $ Matrix fp r c c 0
    matrixScale s (Matrix fp r c _ _) =
        withForeignPtr fp $ \ptr -> do
            mptr <- c_matrix_scale_f64 ptr s
            if mptr == nullPtr
                then error "Matrix scale failed"
                else do
                    fp' <- newForeignPtr c_matrix_finalizer_f64 mptr
                    return $ Matrix fp' r c c 0
    matrixTrace (Matrix fp _ _ _ _) = withForeignPtr fp c_matrix_trace_f64
    matrixNorm (Matrix fp _ _ _ _) = withForeignPtr fp c_matrix_norm_f64

-- ============================================================
-- Construction
-- ============================================================

-- | Create matrix of zeros.
zeros :: MatrixElem a => Int -> Int -> Matrix a
zeros r c = unsafePerformIO $ matrixZeros r c
{-# NOINLINE zeros #-}

-- | Create matrix of ones.
ones :: (Num a, MatrixElem a) => Int -> Int -> Matrix a
ones r c = full r c 1

-- | Create matrix filled with value.
full :: MatrixElem a => Int -> Int -> a -> Matrix a
full r c x = fromList r c (P.replicate (r * c) x)

-- | Create matrix from nested lists.
--
-- >>> fromLists [[1, 2], [3, 4]]
-- Matrix 2x2 [[1, 2], [3, 4]]
fromLists :: MatrixElem a => [[a]] -> Matrix a
fromLists xss =
    let r = P.length xss
        c = if r > 0 then P.length (P.head xss) else 0
    in fromList r c (P.concat xss)

-- | Create matrix from flat list with dimensions.
fromList :: MatrixElem a => Int -> Int -> [a] -> Matrix a
fromList r c xs = unsafePerformIO $ matrixFromList r c xs
{-# NOINLINE fromList #-}

-- | Create matrix from row vectors.
fromRows :: (V.VectorElem a, MatrixElem a) => [V.Vector a] -> Matrix a
fromRows vs =
    let r = P.length vs
        c = if r > 0 then V.length (P.head vs) else 0
    in fromList r c (P.concatMap V.toList vs)

-- | Create matrix from column vectors.
fromCols :: (V.VectorElem a, MatrixElem a) => [V.Vector a] -> Matrix a
fromCols vs = transpose (fromRows vs)

-- | Identity matrix.
--
-- >>> identity 3
-- Matrix 3x3 [[1,0,0], [0,1,0], [0,0,1]]
identity :: MatrixElem a => Int -> Matrix a
identity n = unsafePerformIO $ matrixIdentity n
{-# NOINLINE identity #-}

-- | Identity matrix (alias for identity).
eye :: MatrixElem a => Int -> Matrix a
eye = identity

-- | Diagonal matrix from list.
diag :: (Num a, MatrixElem a) => [a] -> Matrix a
diag xs =
    let n = P.length xs
        indices = [(i, j) | i <- [0..n-1], j <- [0..n-1]]
        elems = [if i == j then xs P.!! i else 0 | (i, j) <- indices]
    in fromList n n elems

-- | Diagonal matrix from vector.
diagFrom :: (Num a, V.VectorElem a, MatrixElem a) => V.Vector a -> Matrix a
diagFrom v = diag (V.toList v)

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
(!) :: MatrixElem a => Matrix a -> (Int, Int) -> a
m ! (i, j) = unsafePerformIO $ matrixGet m i j
{-# NOINLINE (!) #-}

-- | Index into matrix (safe).
(!?) :: MatrixElem a => Matrix a -> (Int, Int) -> Maybe a
m !? (i, j)
    | i < 0 || i >= rows m = Nothing
    | j < 0 || j >= cols m = Nothing
    | otherwise = Just (m ! (i, j))

-- | Extract single row as vector.
row :: (MatrixElem a, V.VectorElem a) => Int -> Matrix a -> V.Vector a
row i m = V.fromList [m ! (i, j) | j <- [0..cols m - 1]]

-- | Extract single column as vector.
col :: (MatrixElem a, V.VectorElem a) => Int -> Matrix a -> V.Vector a
col j m = V.fromList [m ! (i, j) | i <- [0..rows m - 1]]

-- | Get row (alias for row).
getRow :: (MatrixElem a, V.VectorElem a) => Int -> Matrix a -> V.Vector a
getRow = row

-- | Get column (alias for col).
getCol :: (MatrixElem a, V.VectorElem a) => Int -> Matrix a -> V.Vector a
getCol = col

-- | Get main diagonal.
getDiag :: (MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
getDiag m = V.fromList [m ! (i, i) | i <- [0.. P.min (rows m) (cols m) - 1]]

-- ============================================================
-- Slicing
-- ============================================================

-- | Extract submatrix.
--
-- >>> submatrix 0 2 0 2 m  -- 2x2 top-left corner
submatrix :: MatrixElem a => Int -> Int -> Int -> Int -> Matrix a -> Matrix a
submatrix r1 r2 c1 c2 m =
    let newRows = r2 - r1
        newCols = c2 - c1
    in fromList newRows newCols [m ! (i, j) | i <- [r1..r2-1], j <- [c1..c2-1]]

-- | Take first n rows.
takeRows :: MatrixElem a => Int -> Matrix a -> Matrix a
takeRows n m = submatrix 0 n 0 (cols m) m

-- | Drop first n rows.
dropRows :: MatrixElem a => Int -> Matrix a -> Matrix a
dropRows n m = submatrix n (rows m) 0 (cols m) m

-- | Take first n columns.
takeCols :: MatrixElem a => Int -> Matrix a -> Matrix a
takeCols n m = submatrix 0 (rows m) 0 n m

-- | Drop first n columns.
dropCols :: MatrixElem a => Int -> Matrix a -> Matrix a
dropCols n m = submatrix 0 (rows m) n (cols m) m

-- ============================================================
-- Combining
-- ============================================================

-- | Horizontal concatenation.
--
-- >>> a ||| b
-- [a b]
(|||) :: MatrixElem a => Matrix a -> Matrix a -> Matrix a
a ||| b =
    let r = rows a
        c1 = cols a
        c2 = cols b
    in if rows a /= rows b
       then error "Row counts must match for horizontal concatenation"
       else fromList r (c1 + c2)
            [if j < c1 then a ! (i, j) else b ! (i, j - c1)
             | i <- [0..r-1], j <- [0..c1+c2-1]]
infixr 5 |||

-- | Vertical concatenation.
--
-- >>> a --- b
-- [a]
-- [b]
(---) :: MatrixElem a => Matrix a -> Matrix a -> Matrix a
a --- b =
    let r1 = rows a
        r2 = rows b
        c = cols a
    in if cols a /= cols b
       then error "Column counts must match for vertical concatenation"
       else fromList (r1 + r2) c
            [if i < r1 then a ! (i, j) else b ! (i - r1, j)
             | i <- [0..r1+r2-1], j <- [0..c-1]]
infixr 4 ---

-- | Horizontal concatenation of multiple matrices.
hcat :: MatrixElem a => [Matrix a] -> Matrix a
hcat = P.foldl1 (|||)

-- | Vertical concatenation of multiple matrices.
vcat :: MatrixElem a => [Matrix a] -> Matrix a
vcat = P.foldl1 (---)

-- | Create matrix from blocks.
--
-- >>> fromBlocks [[a, b], [c, d]]
fromBlocks :: MatrixElem a => [[Matrix a]] -> Matrix a
fromBlocks blocks = vcat (P.map hcat blocks)

-- ============================================================
-- Element-wise Operations
-- ============================================================

-- | Map function over elements.
map :: (MatrixElem a, MatrixElem b) => (a -> b) -> Matrix a -> Matrix b
map f m = fromList (rows m) (cols m)
    [f (m ! (i, j)) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Map with indices.
imap :: (MatrixElem a, MatrixElem b) => (Int -> Int -> a -> b) -> Matrix a -> Matrix b
imap f m = fromList (rows m) (cols m)
    [f i j (m ! (i, j)) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Zip two matrices.
zipWith :: (MatrixElem a, MatrixElem b, MatrixElem c) => (a -> b -> c) -> Matrix a -> Matrix b -> Matrix c
zipWith f ma mb =
    let r = P.min (rows ma) (rows mb)
        c = P.min (cols ma) (cols mb)
    in fromList r c
       [f (ma ! (i, j)) (mb ! (i, j)) | i <- [0..r-1], j <- [0..c-1]]

-- | Zip with indices.
izipWith :: (MatrixElem a, MatrixElem b, MatrixElem c) => (Int -> Int -> a -> b -> c) -> Matrix a -> Matrix b -> Matrix c
izipWith f ma mb =
    let r = P.min (rows ma) (rows mb)
        c = P.min (cols ma) (cols mb)
    in fromList r c
       [f i j (ma ! (i, j)) (mb ! (i, j)) | i <- [0..r-1], j <- [0..c-1]]

-- ============================================================
-- Matrix Operations
-- ============================================================

-- | Transpose matrix.
--
-- >>> transpose m
-- m^T
transpose :: MatrixElem a => Matrix a -> Matrix a
transpose m = unsafePerformIO $ matrixTranspose m
{-# NOINLINE transpose #-}

-- | Transpose operator.
(^.) :: MatrixElem a => Matrix a -> Matrix a
(^.) = transpose
infixl 8 ^.

-- | Element-wise addition.
(+.) :: (Num a, MatrixElem a) => Matrix a -> Matrix a -> Matrix a
(+.) = zipWith (+)
infixl 6 +.

-- | Element-wise subtraction.
(-.) :: (Num a, MatrixElem a) => Matrix a -> Matrix a -> Matrix a
(-.) = zipWith (-)
infixl 6 -.

-- | Element-wise multiplication (Hadamard product).
(*.) :: (Num a, MatrixElem a) => Matrix a -> Matrix a -> Matrix a
(*.) = zipWith (*)
infixl 7 *.

-- | Element-wise division.
(/.) :: (Fractional a, MatrixElem a) => Matrix a -> Matrix a -> Matrix a
(/.) = zipWith (/)
infixl 7 /.

-- | Scale matrix by scalar.
scale :: MatrixElem a => a -> Matrix a -> Matrix a
scale k m = unsafePerformIO $ matrixScale k m
{-# NOINLINE scale #-}

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
mul :: MatrixElem a => Matrix a -> Matrix a -> Matrix a
mul a b = unsafePerformIO $ matrixMatmul a b
{-# NOINLINE mul #-}

-- | Matrix multiplication operator.
(@@) :: MatrixElem a => Matrix a -> Matrix a -> Matrix a
(@@) = mul
infixl 7 @@

-- | Matrix-vector multiplication.
--
-- >>> mulV m v  -- m @ v
mulV :: (Num a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> V.Vector a
mulV m v =
    let r = rows m
        c = cols m
    in V.fromList [P.sum [m ! (i, j) * (v V.! j) | j <- [0..c-1]] | i <- [0..r-1]]

-- ============================================================
-- Linear Algebra
-- ============================================================

-- | Matrix trace (sum of diagonal).
trace :: MatrixElem a => Matrix a -> a
trace m = unsafePerformIO $ matrixTrace m
{-# NOINLINE trace #-}

-- | Matrix determinant.
--
-- Uses LU decomposition for n>2.
det :: (Fractional a, Ord a, MatrixElem a) => Matrix a -> a
det m
    | rows m /= cols m = error "Determinant requires square matrix"
    | n == 0 = 1
    | n == 1 = m ! (0, 0)
    | n == 2 = m ! (0, 0) * m ! (1, 1) - m ! (0, 1) * m ! (1, 0)
    | otherwise =
        -- Use LU decomposition: det(A) = det(L) * det(U) * det(P)^(-1)
        -- det(L) = 1 (unit lower triangular), det(P) = (-1)^swaps
        let (_, u, _, swaps) = luWithPivot m
            diagProd = P.product [u ! (i, i) | i <- [0..n-1]]
            sign = if even swaps then 1 else (-1)
        in sign * diagProd
  where
    n = rows m

-- | Matrix rank.
--
-- Computed via SVD (count of non-zero singular values).
rank :: (Floating a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> Int
rank m =
    let (_, s, _) = svd m
        tolerance = 1e-10 * V.maximum (V.map P.abs s)
    in V.length (V.filter (\x -> P.abs x > tolerance) s)

-- | Matrix inverse.
--
-- Throws error if matrix is singular.
-- Uses Gauss-Jordan elimination.
inv :: (Fractional a, Ord a, MatrixElem a) => Matrix a -> Matrix a
inv m
    | rows m /= cols m = error "Inverse requires square matrix"
    | otherwise = gaussJordanInverse m

-- | Gauss-Jordan elimination for matrix inverse.
gaussJordanInverse :: (Fractional a, Ord a, MatrixElem a) => Matrix a -> Matrix a
gaussJordanInverse m =
    let n = rows m
        -- Augment matrix [A | I]
        augmented = m ||| identity n
        -- Forward elimination with partial pivoting
        reduced = gaussElim n (2 * n) augmented
        -- Extract right half (inverse)
    in submatrix 0 n n (2 * n) reduced

-- | Gaussian elimination with partial pivoting (modifies to reduced row echelon form).
gaussElim :: (Fractional a, Ord a, MatrixElem a) => Int -> Int -> Matrix a -> Matrix a
gaussElim nRows nCols m = go 0 m
  where
    go col mat
        | col >= nRows = mat
        | otherwise =
            -- Find pivot (max abs value in column)
            let pivotRow = findPivotRow col col mat nRows
                mat1 = swapRows col pivotRow mat
                pivotVal = mat1 ! (col, col)
            in if P.abs pivotVal < 1e-10
               then error "Matrix is singular"
               else
                   -- Scale pivot row
                   let mat2 = scaleRow col (1 / pivotVal) mat1
                       -- Eliminate column
                       mat3 = eliminateColumn col mat2 nRows
                   in go (col + 1) mat3

    findPivotRow col startRow mat maxRow =
        let candidates = [(i, P.abs (mat ! (i, col))) | i <- [startRow..maxRow-1]]
        in fst $ P.maximumBy (\(_, a) (_, b) -> compare a b) candidates

    swapRows r1 r2 mat
        | r1 == r2 = mat
        | otherwise = fromList (rows mat) (cols mat)
            [if i == r1 then mat ! (r2, j)
             else if i == r2 then mat ! (r1, j)
             else mat ! (i, j)
            | i <- [0..rows mat - 1], j <- [0..cols mat - 1]]

    scaleRow r s mat = fromList (rows mat) (cols mat)
        [if i == r then s * mat ! (i, j) else mat ! (i, j)
        | i <- [0..rows mat - 1], j <- [0..cols mat - 1]]

    eliminateColumn col mat maxRow = fromList (rows mat) (cols mat)
        [if i /= col
         then mat ! (i, j) - mat ! (i, col) * mat ! (col, j)
         else mat ! (i, j)
        | i <- [0..rows mat - 1], j <- [0..cols mat - 1]]

-- | Moore-Penrose pseudoinverse.
--
-- A+ = V * S+ * U^T where S+ has reciprocals of non-zero singular values.
pinv :: (Floating a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> Matrix a
pinv m =
    let (u, s, vt) = svd m
        tolerance = 1e-10 * V.maximum (V.map P.abs s)
        sInv = V.map (\x -> if P.abs x > tolerance then 1 / x else 0) s
        -- S+ is diagonal matrix from sInv, dimensions transposed
        k = V.length s
        sInvMat = diag (V.toList sInv)
    in (transpose vt) @@ sInvMat @@ (transpose u)

-- | Solve linear system Ax = b.
--
-- Returns x such that Ax = b.
-- Uses LU decomposition with forward/back substitution.
solve :: (Fractional a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> V.Vector a
solve a b
    | rows a /= cols a = error "solve requires square matrix"
    | rows a /= V.length b = error "solve: dimension mismatch"
    | otherwise =
        let (l, u, p, _) = luWithPivot a
            n = rows a
            -- Apply permutation to b
            pb = V.generate n (\i -> b V.! (permIdx p i))
            -- Forward substitution: Ly = Pb
            y = forwardSubst l pb
            -- Back substitution: Ux = y
        in backSubst u y
  where
    permIdx p i = P.fst $ P.head $ P.filter (\(_, j) -> p ! (i, j) == 1) [(k, k) | k <- [0..rows p - 1]]

-- | Forward substitution for Lx = b (L is lower triangular with unit diagonal).
forwardSubst :: (Fractional a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> V.Vector a
forwardSubst l b = V.generate n solve_i
  where
    n = V.length b
    solve_i i =
        let sum_j = P.sum [l ! (i, j) * (result V.! j) | j <- [0..i-1]]
        in b V.! i - sum_j
    result = forwardSubst l b

-- | Back substitution for Ux = b (U is upper triangular).
backSubst :: (Fractional a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> V.Vector a
backSubst u b = go (n-1) (V.replicate n 0)
  where
    n = V.length b
    go i acc
        | i < 0 = acc
        | otherwise =
            let sum_j = P.sum [u ! (i, j) * (acc V.! j) | j <- [i+1..n-1]]
                xi = (b V.! i - sum_j) / (u ! (i, i))
            in go (i-1) (V.update acc i xi)

-- | Least squares solution.
--
-- Minimizes ||Ax - b||_2.
-- Uses QR decomposition: A = QR, then Rx = Q^T b.
lstsq :: (Floating a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> V.Vector a
lstsq a b =
    let (q, r) = qr a
        qt = transpose q
        qtb = mulV qt b
        -- Solve Rx = Q^T b via back substitution
        n = cols a
    in backSubstRect r qtb n

-- ============================================================
-- Decompositions
-- ============================================================

-- | LU decomposition.
--
-- Returns (L, U, P) where PA = LU.
lu :: (Fractional a, Ord a, MatrixElem a) => Matrix a -> (Matrix a, Matrix a, Matrix a)
lu m =
    let (l, u, p, _) = luWithPivot m
    in (l, u, p)

-- | Internal LU decomposition with pivot count.
luWithPivot :: (Fractional a, Ord a, MatrixElem a) => Matrix a -> (Matrix a, Matrix a, Matrix a, Int)
luWithPivot m
    | rows m /= cols m = error "LU decomposition requires square matrix"
    | otherwise = go 0 (identity n) (identity n) m 0
  where
    n = rows m

    go k l u mat swaps
        | k >= n = (l, mat, u, swaps)  -- u is actually P here, mat is U
        | otherwise =
            -- Find pivot
            let pivotRow = findPivotRow k mat
                (mat', p', newSwaps) =
                    if pivotRow /= k
                    then (swapRows k pivotRow mat, swapRows k pivotRow u, swaps + 1)
                    else (mat, u, swaps)
                pivotVal = mat' ! (k, k)
            in if P.abs pivotVal < 1e-14
               then go (k + 1) l mat' p' newSwaps  -- Skip zero pivot
               else
                   -- Compute multipliers and eliminate
                   let factors = [(i, mat' ! (i, k) / pivotVal) | i <- [k+1..n-1]]
                       mat'' = eliminateBelow k factors mat'
                       l' = setFactors k factors l
                   in go (k + 1) l' mat'' p' newSwaps

    findPivotRow k mat =
        let candidates = [(i, P.abs (mat ! (i, k))) | i <- [k..n-1]]
        in fst $ P.maximumBy (\(_, a) (_, b) -> compare a b) candidates

    swapRows r1 r2 mat = fromList n n
        [if i == r1 then mat ! (r2, j)
         else if i == r2 then mat ! (r1, j)
         else mat ! (i, j)
        | i <- [0..n-1], j <- [0..n-1]]

    eliminateBelow k factors mat = fromList n n
        [let factor = P.lookup i factors
         in case factor of
             Just f -> if j >= k then mat ! (i, j) - f * mat ! (k, j) else mat ! (i, j)
             Nothing -> mat ! (i, j)
        | i <- [0..n-1], j <- [0..n-1]]

    setFactors k factors l = fromList n n
        [if i > k && j == k
         then case P.lookup i factors of
             Just f -> f
             Nothing -> 0
         else l ! (i, j)
        | i <- [0..n-1], j <- [0..n-1]]

-- | QR decomposition.
--
-- Returns (Q, R) where A = QR.
-- Uses modified Gram-Schmidt orthogonalization.
qr :: (Floating a, Ord a, MatrixElem a) => Matrix a -> (Matrix a, Matrix a)
qr m = (q, r)
  where
    nRows = rows m
    nCols = cols m
    k = P.min nRows nCols

    -- Gram-Schmidt process
    (qCols, rData) = gramSchmidt 0 [] []

    gramSchmidt j qAcc rAcc
        | j >= k = (qAcc, rAcc)
        | otherwise =
            let colJ = [m ! (i, j) | i <- [0..nRows-1]]
                -- Subtract projections onto previous q vectors
                (v, rEntries) = subtractProjections colJ qAcc 0 []
                -- Normalize
                normV = P.sqrt (P.sum (P.map (^(2::Int)) v))
                qj = if normV > 1e-14
                     then P.map (/ normV) v
                     else P.replicate nRows 0
                rjj = normV
            in gramSchmidt (j + 1) (qAcc ++ [qj]) (rAcc ++ [(j, rEntries ++ [(j, rjj)])])

    subtractProjections v [] _ acc = (v, acc)
    subtractProjections v (qi:qis) i acc =
        let rij = P.sum (P.zipWith (*) v qi)
            v' = P.zipWith (\vk qik -> vk - rij * qik) v qi
        in subtractProjections v' qis (i + 1) (acc ++ [(i, rij)])

    q = fromList nRows k [qCols !! j !! i | i <- [0..nRows-1], j <- [0..k-1]]
    r = fromList k nCols
        [case P.lookup j rData of
             Just entries -> case P.lookup i entries of
                 Just rij -> rij
                 Nothing -> 0
             Nothing -> 0
        | i <- [0..k-1], j <- [0..nCols-1]]

-- | Singular value decomposition.
--
-- Returns (U, S, V^T) where A = U * diag(S) * V^T.
-- Uses power iteration method for computing dominant singular vectors.
svd :: (Floating a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> (Matrix a, V.Vector a, Matrix a)
svd m =
    let nRows = rows m
        nCols = cols m
        k = P.min nRows nCols
        -- Compute singular values and vectors iteratively
        ata = (transpose m) @@ m
        aat = m @@ (transpose m)
        -- Get singular values from eigenvalues of A^T A
        (singVals, vVecs) = powerIterationMultiple ata k
        -- Compute U from AV/sigma
        uCols = P.zipWith (\sigma vCol ->
            if sigma > 1e-14
            then V.map (/ sigma) (mulV m vCol)
            else V.replicate nRows 0) singVals vVecs
        uMat = fromCols uCols
        vMat = fromCols vVecs
    in (uMat, V.fromList singVals, transpose vMat)

-- | Power iteration for multiple eigenvectors.
powerIterationMultiple :: (Floating a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> Int -> ([a], [V.Vector a])
powerIterationMultiple m k = go k [] []
  where
    n = rows m
    maxIter = 100

    go 0 vals vecs = (P.reverse vals, P.reverse vecs)
    go remaining vals vecs =
        let -- Start with random-ish vector
            v0 = V.fromList [1 / P.sqrt (fromIntegral n) | _ <- [1..n]]
            -- Orthogonalize against existing eigenvectors
            v0Orth = orthogonalize v0 vecs
            -- Power iteration
            (lambda, v) = powerIter v0Orth maxIter
            sigma = P.sqrt (P.abs lambda)
        in go (remaining - 1) (sigma : vals) (v : vecs)

    powerIter v 0 = (0, v)
    powerIter v iter =
        let av = mulV m v
            norm_av = V.norm av
            v' = if norm_av > 1e-14 then V.map (/ norm_av) av else v
            lambda = V.dot (mulV m v') v'
        in if iter <= 1 || norm_av < 1e-14
           then (lambda, v')
           else powerIter v' (iter - 1)

    orthogonalize v [] = v
    orthogonalize v (qi:qis) =
        let proj = V.dot v qi
            v' = V.zipWith (\vi qii -> vi - proj * qii) v qi
            normV = V.norm v'
            v'' = if normV > 1e-14 then V.map (/ normV) v' else v'
        in orthogonalize v'' qis

-- | Cholesky decomposition.
--
-- Returns L where A = LL^T.
-- Requires A to be symmetric positive definite.
cholesky :: (Floating a, Ord a, MatrixElem a) => Matrix a -> Matrix a
cholesky m
    | rows m /= cols m = error "Cholesky requires square matrix"
    | otherwise = fromList n n [computeL i j | i <- [0..n-1], j <- [0..n-1]]
  where
    n = rows m

    computeL i j
        | j > i = 0  -- Upper triangle is zero
        | i == j = -- Diagonal
            let sumSq = P.sum [lij i k ^ (2::Int) | k <- [0..j-1]]
                val = m ! (i, i) - sumSq
            in if val <= 0
               then error "Cholesky: matrix is not positive definite"
               else P.sqrt val
        | otherwise = -- Lower triangle
            let sumProd = P.sum [lij i k * lij j k | k <- [0..j-1]]
                ljj = lij j j
            in if ljj == 0
               then error "Cholesky: zero diagonal"
               else (m ! (i, j) - sumProd) / ljj

    -- Memoized L values (computed on demand)
    lMatrix = fromList n n [computeL i j | i <- [0..n-1], j <- [0..n-1]]
    lij i j = lMatrix ! (i, j)

-- | Eigendecomposition.
--
-- Returns (eigenvalues, eigenvectors).
-- Eigenvectors are columns of the matrix.
-- Uses power iteration for dominant eigenvalues.
eig :: (Floating a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> (V.Vector a, Matrix a)
eig m
    | rows m /= cols m = error "Eigendecomposition requires square matrix"
    | otherwise =
        let n = rows m
            (vals, vecs) = powerIterationMultiple m n
        in (V.fromList vals, fromCols vecs)

-- | Eigenvalues only.
eigvals :: (Floating a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
eigvals m = fst (eig m)

-- | Back substitution for rectangular upper triangular system.
backSubstRect :: (Fractional a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> Int -> V.Vector a
backSubstRect r b n = go (n-1) (V.replicate n 0)
  where
    go i acc
        | i < 0 = acc
        | otherwise =
            let rii = r ! (i, i)
                sum_j = P.sum [r ! (i, j) * (acc V.! j) | j <- [i+1..n-1]]
                xi = if P.abs rii > 1e-14
                     then (b V.! i - sum_j) / rii
                     else 0
            in go (i-1) (V.update acc i xi)

-- ============================================================
-- Norms
-- ============================================================

-- | 1-norm (maximum column sum).
norm1 :: (Num a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> a
norm1 m = P.maximum [V.sum (V.map P.abs (col j m)) | j <- [0..cols m - 1]]

-- | 2-norm (spectral norm, largest singular value).
-- Note: Requires SVD for correct implementation, using Frobenius as approximation
norm2 :: MatrixElem a => Matrix a -> a
norm2 m = unsafePerformIO $ matrixNorm m
{-# NOINLINE norm2 #-}

-- | Infinity norm (maximum row sum).
normInf :: (Num a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> a
normInf m = P.maximum [V.sum (V.map P.abs (row i m)) | i <- [0..rows m - 1]]

-- | Frobenius norm (sqrt of sum of squares).
normFrob :: (Floating a, MatrixElem a) => Matrix a -> a
normFrob m = P.sqrt (P.sum [m ! (i, j) ^ (2 :: Int) | i <- [0..rows m - 1], j <- [0..cols m - 1]])

-- ============================================================
-- Folds
-- ============================================================

-- | Sum of all elements.
sum :: (Num a, MatrixElem a) => Matrix a -> a
sum m = P.sum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Product of all elements.
product :: (Num a, MatrixElem a) => Matrix a -> a
product m = P.product [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Maximum element.
maximum :: (Ord a, MatrixElem a) => Matrix a -> a
maximum m = P.maximum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Minimum element.
minimum :: (Ord a, MatrixElem a) => Matrix a -> a
minimum m = P.minimum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Sum each row.
sumRows :: (Num a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
sumRows m = V.generate (rows m) (\i -> V.sum (row i m))

-- | Sum each column.
sumCols :: (Num a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
sumCols m = V.generate (cols m) (\j -> V.sum (col j m))

-- | Mean of each row.
meanRows :: (Fractional a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
meanRows m = V.map (/ P.fromIntegral (cols m)) (sumRows m)

-- | Mean of each column.
meanCols :: (Fractional a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
meanCols m = V.map (/ P.fromIntegral (rows m)) (sumCols m)

-- ============================================================
-- Conversion
-- ============================================================

-- | Convert to nested lists.
toLists :: MatrixElem a => Matrix a -> [[a]]
toLists m = [[m ! (i, j) | j <- [0..cols m - 1]] | i <- [0..rows m - 1]]

-- | Convert to flat list (row-major).
toList :: MatrixElem a => Matrix a -> [a]
toList = P.concat . toLists

-- | Flatten to vector.
flatten :: (MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
flatten m = V.fromList (toList m)

-- | View vector as 1-column matrix.
asColumn :: (V.VectorElem a, MatrixElem a) => V.Vector a -> Matrix a
asColumn v = fromList (V.length v) 1 (V.toList v)

-- | View vector as 1-row matrix.
asVector :: (V.VectorElem a, MatrixElem a) => V.Vector a -> Matrix a
asVector v = fromList 1 (V.length v) (V.toList v)
