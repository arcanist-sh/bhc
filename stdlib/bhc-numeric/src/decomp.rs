//! Matrix decompositions
//!
//! This module provides numerical matrix decomposition algorithms:
//!
//! - **LU Decomposition**: Factors A = PLU where P is a permutation matrix,
//!   L is lower triangular, and U is upper triangular
//! - **QR Decomposition**: Factors A = QR where Q is orthogonal and R is
//!   upper triangular (Householder method)
//! - **Cholesky Decomposition**: Factors A = LL^T for symmetric positive definite A
//!
//! # Numerical Stability
//!
//! All algorithms use partial pivoting or other stabilization techniques
//! to ensure numerical accuracy. Singular or near-singular matrices are
//! detected and reported via the `DecompError` type.
//!
//! # Example
//!
//! ```ignore
//! use bhc_numeric::decomp::{lu_decompose, qr_decompose};
//! use bhc_numeric::matrix::Matrix;
//!
//! let a = Matrix::from_data(3, 3, vec![
//!     2.0, -1.0, 0.0,
//!     -1.0, 2.0, -1.0,
//!     0.0, -1.0, 2.0,
//! ]);
//!
//! let lu = lu_decompose(&a).unwrap();
//! let qr = qr_decompose(&a);
//! ```

use crate::matrix::Matrix;
use std::fmt;

// ============================================================
// Error Types
// ============================================================

/// Errors that can occur during matrix decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum DecompError {
    /// Matrix is singular (has zero or near-zero pivot).
    Singular { pivot_index: usize, value: f64 },
    /// Matrix is not square when a square matrix is required.
    NotSquare { rows: usize, cols: usize },
    /// Matrix is not symmetric positive definite (for Cholesky).
    NotPositiveDefinite { index: usize },
    /// Dimensions are incompatible.
    DimensionMismatch { expected: (usize, usize), got: (usize, usize) },
}

impl fmt::Display for DecompError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecompError::Singular { pivot_index, value } => {
                write!(f, "Matrix is singular at pivot {}: value = {:.2e}", pivot_index, value)
            }
            DecompError::NotSquare { rows, cols } => {
                write!(f, "Matrix must be square, got {}x{}", rows, cols)
            }
            DecompError::NotPositiveDefinite { index } => {
                write!(f, "Matrix is not positive definite at index {}", index)
            }
            DecompError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {:?}, got {:?}", expected, got)
            }
        }
    }
}

impl std::error::Error for DecompError {}

// ============================================================
// LU Decomposition
// ============================================================

/// Result of LU decomposition.
///
/// Contains the factorization A = PLU where:
/// - P is a permutation matrix (stored as a pivot array)
/// - L is unit lower triangular (diagonal is 1)
/// - U is upper triangular
///
/// L and U are stored in-place in a single matrix with the following layout:
/// ```text
/// [U00 U01 U02]   L stored below diagonal
/// [L10 U11 U12]   U stored on and above diagonal
/// [L20 L21 U22]   L diagonal is implicitly 1
/// ```
#[derive(Debug, Clone)]
pub struct LuResult {
    /// Combined L and U matrix (L below diagonal, U on and above diagonal).
    pub lu: Matrix<f64>,
    /// Pivot indices: row i was swapped with row pivot[i].
    pub pivot: Vec<usize>,
    /// Number of row swaps (parity of permutation).
    pub num_swaps: usize,
}

impl LuResult {
    /// Extract the L matrix (unit lower triangular).
    pub fn l(&self) -> Matrix<f64> {
        let n = self.lu.rows();
        let mut l = Matrix::identity(n);
        for i in 1..n {
            for j in 0..i {
                *l.get_mut(i, j).unwrap() = self.lu[(i, j)];
            }
        }
        l
    }

    /// Extract the U matrix (upper triangular).
    pub fn u(&self) -> Matrix<f64> {
        let n = self.lu.rows();
        let mut u = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                *u.get_mut(i, j).unwrap() = self.lu[(i, j)];
            }
        }
        u
    }

    /// Get the permutation matrix P.
    pub fn p(&self) -> Matrix<f64> {
        let n = self.pivot.len();
        let mut p = Matrix::identity(n);
        for i in 0..n {
            if self.pivot[i] != i {
                p.swap_rows(i, self.pivot[i]);
            }
        }
        p
    }

    /// Compute the determinant from LU decomposition.
    ///
    /// det(A) = det(P) * det(L) * det(U) = (-1)^swaps * 1 * prod(U_ii)
    pub fn determinant(&self) -> f64 {
        let n = self.lu.rows();
        let sign = if self.num_swaps % 2 == 0 { 1.0 } else { -1.0 };
        let mut det = sign;
        for i in 0..n {
            det *= self.lu[(i, i)];
        }
        det
    }

    /// Solve Ax = b using the LU decomposition.
    ///
    /// Solves in two steps:
    /// 1. Ly = Pb (forward substitution)
    /// 2. Ux = y (backward substitution)
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.lu.rows();
        assert_eq!(b.len(), n, "b must have length equal to matrix size");

        // Apply permutation: pb = P * b
        let mut pb = b.to_vec();
        for i in 0..n {
            if self.pivot[i] != i {
                pb.swap(i, self.pivot[i]);
            }
        }

        // Forward substitution: Ly = pb
        let mut y = pb;
        for i in 0..n {
            for j in 0..i {
                y[i] -= self.lu[(i, j)] * y[j];
            }
        }

        // Backward substitution: Ux = y
        let mut x = y;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= self.lu[(i, j)] * x[j];
            }
            x[i] /= self.lu[(i, i)];
        }

        x
    }
}

/// Perform LU decomposition with partial pivoting.
///
/// Factors A = PLU where:
/// - P is a permutation matrix
/// - L is unit lower triangular (1s on diagonal)
/// - U is upper triangular
///
/// # Algorithm
///
/// Uses Doolittle's algorithm with partial pivoting for numerical stability.
/// Partial pivoting swaps rows to put the largest element on the diagonal,
/// which helps avoid numerical issues with small pivots.
///
/// # Errors
///
/// Returns `DecompError::Singular` if the matrix is singular (has a zero pivot).
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::decomp::lu_decompose;
/// use bhc_numeric::matrix::Matrix;
///
/// let a = Matrix::from_data(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
/// let lu = lu_decompose(&a).unwrap();
///
/// // Solve Ax = b
/// let b = vec![1.0, 2.0];
/// let x = lu.solve(&b);
/// ```
pub fn lu_decompose(a: &Matrix<f64>) -> Result<LuResult, DecompError> {
    if !a.is_square() {
        return Err(DecompError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    let mut lu = a.clone();
    // pivot[k] records which row was swapped with row k during step k
    let mut pivot = (0..n).collect::<Vec<_>>();
    let mut num_swaps = 0;

    // Tolerance for detecting singularity
    const EPSILON: f64 = 1e-14;

    for k in 0..n {
        // Find pivot: largest absolute value in column k at or below row k
        let mut max_val = lu[(k, k)].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[(i, k)].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singularity
        if max_val < EPSILON {
            return Err(DecompError::Singular {
                pivot_index: k,
                value: max_val,
            });
        }

        // Record which row is swapped with row k
        pivot[k] = max_row;

        // Swap rows if needed
        if max_row != k {
            lu.swap_rows(k, max_row);
            num_swaps += 1;
        }

        // Perform elimination
        let pivot_val = lu[(k, k)];
        for i in (k + 1)..n {
            // Compute multiplier
            let mult = lu[(i, k)] / pivot_val;
            *lu.get_mut(i, k).unwrap() = mult;

            // Update row i
            for j in (k + 1)..n {
                let update = mult * lu[(k, j)];
                *lu.get_mut(i, j).unwrap() -= update;
            }
        }
    }

    Ok(LuResult {
        lu,
        pivot,
        num_swaps,
    })
}

// ============================================================
// QR Decomposition
// ============================================================

/// Result of QR decomposition.
///
/// Contains the factorization A = QR where:
/// - Q is an orthogonal matrix (Q^T * Q = I)
/// - R is upper triangular
#[derive(Debug, Clone)]
pub struct QrResult {
    /// Orthogonal matrix Q (m x m or m x n depending on thin vs full).
    pub q: Matrix<f64>,
    /// Upper triangular matrix R.
    pub r: Matrix<f64>,
}

impl QrResult {
    /// Solve the least squares problem min ||Ax - b||_2.
    ///
    /// For overdetermined systems (m > n), this gives the least squares solution.
    /// For square systems, this gives the exact solution.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let m = self.q.rows();
        let n = self.r.cols();
        assert_eq!(b.len(), m, "b must have length equal to number of rows");

        // y = Q^T * b
        let mut y = vec![0.0; n];
        for j in 0..n {
            for i in 0..m {
                y[j] += self.q[(i, j)] * b[i];
            }
        }

        // Solve Rx = y by back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                x[i] -= self.r[(i, j)] * x[j];
            }
            x[i] /= self.r[(i, i)];
        }

        x
    }

    /// Compute the pseudoinverse A^+ using QR decomposition.
    ///
    /// For full-rank matrices: A^+ = R^-1 * Q^T
    pub fn pseudoinverse(&self) -> Matrix<f64> {
        let n = self.r.cols();
        let m = self.q.rows();

        // Compute R^-1 by back substitution for each column
        let mut r_inv = Matrix::zeros(n, n);
        for col in 0..n {
            let mut e = vec![0.0; n];
            e[col] = 1.0;

            // Solve R * x = e_col
            for i in (0..n).rev() {
                let mut val = e[i];
                for j in (i + 1)..n {
                    val -= self.r[(i, j)] * r_inv[(j, col)];
                }
                *r_inv.get_mut(i, col).unwrap() = val / self.r[(i, i)];
            }
        }

        // A^+ = R^-1 * Q^T
        let mut result = Matrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                for k in 0..n {
                    *result.get_mut(i, j).unwrap() += r_inv[(i, k)] * self.q[(j, k)];
                }
            }
        }

        result
    }
}

/// Perform QR decomposition using Householder reflections.
///
/// Factors A = QR where:
/// - Q is an orthogonal matrix (m x m)
/// - R is upper triangular (m x n)
///
/// # Algorithm
///
/// Uses Householder reflections, which are numerically stable and efficient.
/// Each step creates a Householder matrix H_k that zeros out the subdiagonal
/// elements in column k.
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::decomp::qr_decompose;
/// use bhc_numeric::matrix::Matrix;
///
/// let a = Matrix::from_data(3, 3, vec![
///     12.0, -51.0, 4.0,
///     6.0, 167.0, -68.0,
///     -4.0, 24.0, -41.0,
/// ]);
/// let qr = qr_decompose(&a);
///
/// // Q * R should equal A
/// let reconstructed = qr.q.matmul(&qr.r).unwrap();
/// ```
pub fn qr_decompose(a: &Matrix<f64>) -> QrResult {
    let m = a.rows();
    let n = a.cols();
    let k = m.min(n);

    let mut q = Matrix::identity(m);
    let mut r = a.clone();

    for col in 0..k {
        // Extract column vector below diagonal
        let mut x = vec![0.0; m - col];
        for i in col..m {
            x[i - col] = r[(i, col)];
        }

        // Compute Householder vector
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_x < 1e-14 {
            continue; // Skip if column is already zero
        }

        // v = x + sign(x[0]) * ||x|| * e_1
        x[0] += x[0].signum() * norm_x;
        let norm_v: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

        if norm_v < 1e-14 {
            continue;
        }

        // Normalize v
        for v in &mut x {
            *v /= norm_v;
        }

        // Apply H = I - 2*v*v^T to R (from left)
        // H * R = R - 2 * v * (v^T * R)
        for j in col..n {
            // Compute v^T * R[:, j]
            let mut dot = 0.0;
            for i in col..m {
                dot += x[i - col] * r[(i, j)];
            }
            // Update R[:, j] = R[:, j] - 2 * dot * v
            for i in col..m {
                *r.get_mut(i, j).unwrap() -= 2.0 * dot * x[i - col];
            }
        }

        // Apply H to Q (from right)
        // Q * H = Q - 2 * (Q * v) * v^T
        for i in 0..m {
            // Compute (Q * v)[i] = sum_j Q[i,j] * v[j]
            let mut qv = 0.0;
            for j in col..m {
                qv += q[(i, j)] * x[j - col];
            }
            // Update Q[i, :] = Q[i, :] - 2 * qv * v^T
            for j in col..m {
                *q.get_mut(i, j).unwrap() -= 2.0 * qv * x[j - col];
            }
        }
    }

    QrResult { q, r }
}

// ============================================================
// Cholesky Decomposition
// ============================================================

/// Result of Cholesky decomposition.
///
/// Contains the factorization A = LL^T where L is lower triangular.
#[derive(Debug, Clone)]
pub struct CholeskyResult {
    /// Lower triangular matrix L.
    pub l: Matrix<f64>,
}

impl CholeskyResult {
    /// Solve Ax = b using the Cholesky decomposition.
    ///
    /// Solves in two steps:
    /// 1. Ly = b (forward substitution)
    /// 2. L^T x = y (backward substitution)
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.l.rows();
        assert_eq!(b.len(), n, "b must have length equal to matrix size");

        // Forward substitution: Ly = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = b[i];
            for j in 0..i {
                y[i] -= self.l[(i, j)] * y[j];
            }
            y[i] /= self.l[(i, i)];
        }

        // Backward substitution: L^T x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                x[i] -= self.l[(j, i)] * x[j];
            }
            x[i] /= self.l[(i, i)];
        }

        x
    }

    /// Compute the determinant using Cholesky decomposition.
    ///
    /// det(A) = det(L)^2 = (prod L_ii)^2
    pub fn determinant(&self) -> f64 {
        let n = self.l.rows();
        let mut det_l = 1.0;
        for i in 0..n {
            det_l *= self.l[(i, i)];
        }
        det_l * det_l
    }
}

/// Perform Cholesky decomposition for symmetric positive definite matrices.
///
/// Factors A = LL^T where L is lower triangular.
///
/// # Requirements
///
/// - Matrix must be square
/// - Matrix must be symmetric (A = A^T)
/// - Matrix must be positive definite (all eigenvalues > 0)
///
/// # Errors
///
/// Returns `DecompError::NotSquare` if the matrix is not square.
/// Returns `DecompError::NotPositiveDefinite` if a negative value is encountered
/// during decomposition (indicating the matrix is not positive definite).
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::decomp::cholesky_decompose;
/// use bhc_numeric::matrix::Matrix;
///
/// // Symmetric positive definite matrix
/// let a = Matrix::from_data(3, 3, vec![
///     4.0, 2.0, 2.0,
///     2.0, 10.0, 7.0,
///     2.0, 7.0, 21.0,
/// ]);
/// let chol = cholesky_decompose(&a).unwrap();
/// ```
pub fn cholesky_decompose(a: &Matrix<f64>) -> Result<CholeskyResult, DecompError> {
    if !a.is_square() {
        return Err(DecompError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    let mut l = Matrix::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;

            if i == j {
                // Diagonal elements
                for k in 0..j {
                    sum += l[(j, k)] * l[(j, k)];
                }
                let diag = a[(j, j)] - sum;
                if diag <= 0.0 {
                    return Err(DecompError::NotPositiveDefinite { index: j });
                }
                *l.get_mut(j, j).unwrap() = diag.sqrt();
            } else {
                // Off-diagonal elements
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }
                *l.get_mut(i, j).unwrap() = (a[(i, j)] - sum) / l[(j, j)];
            }
        }
    }

    Ok(CholeskyResult { l })
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // LU Tests

    #[test]
    fn test_lu_basic() {
        let a = Matrix::from_data(3, 3, vec![
            2.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 2.0,
        ]);
        let lu = lu_decompose(&a).unwrap();

        // Verify P*L*U = A
        let p = lu.p();
        let l = lu.l();
        let u = lu.u();
        let plu = p.matmul(&l).unwrap().matmul(&u).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(plu[(i, j)], a[(i, j)], 1e-10),
                    "PLU[{},{}] = {} != A[{},{}] = {}",
                    i, j, plu[(i, j)], i, j, a[(i, j)]);
            }
        }
    }

    #[test]
    fn test_lu_solve() {
        let a = Matrix::from_data(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let lu = lu_decompose(&a).unwrap();

        let b = vec![10.0, 12.0];
        let x = lu.solve(&b);

        // Verify A*x = b
        let ax = vec![
            a[(0, 0)] * x[0] + a[(0, 1)] * x[1],
            a[(1, 0)] * x[0] + a[(1, 1)] * x[1],
        ];

        assert!(approx_eq(ax[0], b[0], 1e-10));
        assert!(approx_eq(ax[1], b[1], 1e-10));
    }

    #[test]
    fn test_lu_determinant() {
        let a = Matrix::from_data(2, 2, vec![3.0, 8.0, 4.0, 6.0]);
        let lu = lu_decompose(&a).unwrap();
        let det = lu.determinant();

        // det = 3*6 - 8*4 = 18 - 32 = -14
        assert!(approx_eq(det, -14.0, 1e-10));
    }

    #[test]
    fn test_lu_singular() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 2.0, 4.0]); // Singular
        let result = lu_decompose(&a);
        assert!(matches!(result, Err(DecompError::Singular { .. })));
    }

    // QR Tests

    #[test]
    fn test_qr_basic() {
        let a = Matrix::from_data(3, 3, vec![
            12.0, -51.0, 4.0,
            6.0, 167.0, -68.0,
            -4.0, 24.0, -41.0,
        ]);
        let qr = qr_decompose(&a);

        // Verify Q*R = A
        let qra = qr.q.matmul(&qr.r).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(qra[(i, j)], a[(i, j)], 1e-10),
                    "QR[{},{}] = {} != A[{},{}] = {}",
                    i, j, qra[(i, j)], i, j, a[(i, j)]);
            }
        }
    }

    #[test]
    fn test_qr_orthogonal() {
        let a = Matrix::from_data(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ]);
        let qr = qr_decompose(&a);

        // Verify Q^T * Q = I
        let qt = qr.q.transpose();
        let qtq = qt.matmul(&qr.q).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(qtq[(i, j)], expected, 1e-10),
                    "Q^T*Q[{},{}] = {} != {}",
                    i, j, qtq[(i, j)], expected);
            }
        }
    }

    #[test]
    fn test_qr_upper_triangular() {
        let a = Matrix::from_data(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ]);
        let qr = qr_decompose(&a);

        // Verify R is upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(approx_eq(qr.r[(i, j)], 0.0, 1e-10),
                    "R[{},{}] = {} should be 0",
                    i, j, qr.r[(i, j)]);
            }
        }
    }

    #[test]
    fn test_qr_solve() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let qr = qr_decompose(&a);

        let b = vec![5.0, 11.0];
        let x = qr.solve(&b);

        // Verify A*x = b
        let ax = vec![
            a[(0, 0)] * x[0] + a[(0, 1)] * x[1],
            a[(1, 0)] * x[0] + a[(1, 1)] * x[1],
        ];

        assert!(approx_eq(ax[0], b[0], 1e-10));
        assert!(approx_eq(ax[1], b[1], 1e-10));
    }

    // Cholesky Tests

    #[test]
    fn test_cholesky_basic() {
        let a = Matrix::from_data(3, 3, vec![
            4.0, 2.0, 2.0,
            2.0, 10.0, 7.0,
            2.0, 7.0, 21.0,
        ]);
        let chol = cholesky_decompose(&a).unwrap();

        // Verify L * L^T = A
        let lt = chol.l.transpose();
        let llt = chol.l.matmul(&lt).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(llt[(i, j)], a[(i, j)], 1e-10),
                    "L*L^T[{},{}] = {} != A[{},{}] = {}",
                    i, j, llt[(i, j)], i, j, a[(i, j)]);
            }
        }
    }

    #[test]
    fn test_cholesky_solve() {
        let a = Matrix::from_data(2, 2, vec![4.0, 2.0, 2.0, 5.0]);
        let chol = cholesky_decompose(&a).unwrap();

        let b = vec![4.0, 3.0];
        let x = chol.solve(&b);

        // Verify A*x = b
        let ax = vec![
            a[(0, 0)] * x[0] + a[(0, 1)] * x[1],
            a[(1, 0)] * x[0] + a[(1, 1)] * x[1],
        ];

        assert!(approx_eq(ax[0], b[0], 1e-10));
        assert!(approx_eq(ax[1], b[1], 1e-10));
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // This matrix is not positive definite
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        let result = cholesky_decompose(&a);
        assert!(matches!(result, Err(DecompError::NotPositiveDefinite { .. })));
    }

    #[test]
    fn test_cholesky_determinant() {
        let a = Matrix::from_data(2, 2, vec![4.0, 2.0, 2.0, 5.0]);
        let chol = cholesky_decompose(&a).unwrap();
        let det = chol.determinant();

        // det = 4*5 - 2*2 = 16
        assert!(approx_eq(det, 16.0, 1e-10));
    }
}
