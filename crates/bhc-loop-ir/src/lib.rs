//! # BHC Loop IR
//!
//! This crate defines the Loop Intermediate Representation for the Basel
//! Haskell Compiler. Loop IR makes iteration structure explicit and is the
//! target for vectorization and low-level optimization.
//!
//! ## Overview
//!
//! Loop IR is the lowest-level IR before code generation. It provides:
//!
//! - **Explicit iteration**: Loops with bounds and strides
//! - **Vectorization information**: Which loops can be SIMD-ized
//! - **Parallelization hints**: Which loops can run in parallel
//! - **Memory access patterns**: For cache optimization
//!
//! ## IR Pipeline Position
//!
//! ```text
//! Source Code
//!     |
//!     v
//! [Parse/AST]
//!     |
//!     v
//! [HIR]
//!     |
//!     v
//! [Core IR]
//!     |
//!     v
//! [Tensor IR]  <- High-level tensor operations
//!     |
//!     v
//! [Loop IR]    <- This crate: explicit iteration
//!     |
//!     v
//! [Codegen]    <- LLVM IR / Native code
//! ```
//!
//! ## Key Transformations
//!
//! Loop IR supports several important optimizations:
//!
//! 1. **Loop tiling**: Break loops into cache-friendly tiles
//! 2. **Vectorization**: Convert scalar operations to SIMD
//! 3. **Parallelization**: Mark loops for parallel execution
//! 4. **Interchange**: Reorder loops for better memory access
//! 5. **Unrolling**: Reduce loop overhead
//!
//! ## Main Types
//!
//! - [`LoopIR`]: The top-level IR structure
//! - [`Loop`]: A single loop with bounds and body
//! - [`Stmt`]: Statements within loop bodies
//! - [`Value`]: SSA values (registers)
//! - [`MemRef`]: Memory references with access patterns
//!
//! ## See Also
//!
//! - `bhc-tensor-ir`: Tensor IR that lowers to Loop IR
//! - `bhc-codegen`: Code generation from Loop IR
//! - H26-SPEC Section 7: Tensor Model (lowering)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_tensor_ir::{AllocRegion, BufferId, DType, Dim, TensorMeta};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A unique identifier for values (SSA registers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueId(u32);

impl Idx for ValueId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A unique identifier for loops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LoopId(u32);

impl Idx for LoopId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A unique identifier for basic blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(u32);

impl Idx for BlockId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// The main Loop IR structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopIR {
    /// Function name.
    pub name: Symbol,
    /// Function parameters.
    pub params: Vec<Param>,
    /// Return type.
    pub return_ty: LoopType,
    /// The body (list of statements and loops).
    pub body: Body,
    /// Memory allocations.
    pub allocs: Vec<Alloc>,
    /// Loop metadata for optimization.
    pub loop_info: Vec<LoopMetadata>,
}

/// A function parameter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Param {
    /// Parameter name.
    pub name: Symbol,
    /// Parameter type.
    pub ty: LoopType,
    /// Whether this is a pointer to memory.
    pub is_ptr: bool,
}

/// Types in Loop IR.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoopType {
    /// Void (no value).
    Void,
    /// Scalar type.
    Scalar(ScalarType),
    /// Vector type (SIMD).
    Vector(ScalarType, u8),
    /// Pointer to memory.
    Ptr(Box<LoopType>),
}

impl LoopType {
    /// Returns the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Void => 0,
            Self::Scalar(s) => s.size_bytes(),
            Self::Vector(s, width) => s.size_bytes() * (*width as usize),
            Self::Ptr(_) => 8, // Assuming 64-bit pointers
        }
    }

    /// Returns true if this is a void type.
    #[must_use]
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Void)
    }

    /// Returns true if this is a vector type.
    #[must_use]
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_, _))
    }
}

/// Scalar types in Loop IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    /// Boolean.
    Bool,
    /// Signed integer with bit width.
    Int(u8),
    /// Unsigned integer with bit width.
    UInt(u8),
    /// Floating point with bit width.
    Float(u8),
}

impl ScalarType {
    /// Returns the size in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Int(bits) | Self::UInt(bits) | Self::Float(bits) => (bits as usize + 7) / 8,
        }
    }

    /// Converts from tensor DType.
    #[must_use]
    pub fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Bool => Self::Bool,
            DType::Int8 => Self::Int(8),
            DType::Int16 => Self::Int(16),
            DType::Int32 => Self::Int(32),
            DType::Int64 => Self::Int(64),
            DType::UInt8 => Self::UInt(8),
            DType::UInt16 => Self::UInt(16),
            DType::UInt32 => Self::UInt(32),
            DType::UInt64 => Self::UInt(64),
            DType::Float16 | DType::BFloat16 => Self::Float(16),
            DType::Float32 => Self::Float(32),
            DType::Float64 => Self::Float(64),
            DType::Complex64 => Self::Float(32), // Represented as pair
            DType::Complex128 => Self::Float(64),
        }
    }
}

/// A memory allocation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alloc {
    /// Buffer identifier.
    pub buffer: BufferId,
    /// Name for debugging.
    pub name: Symbol,
    /// Element type.
    pub elem_ty: ScalarType,
    /// Total size in elements.
    pub size: AllocSize,
    /// Alignment in bytes.
    pub alignment: usize,
    /// Allocation region.
    pub region: AllocRegion,
}

/// Size of an allocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocSize {
    /// Statically known size.
    Static(usize),
    /// Dynamic size (computed at runtime).
    Dynamic(ValueId),
}

/// The body of a function or loop.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Body {
    /// Statements in execution order.
    pub stmts: Vec<Stmt>,
}

impl Body {
    /// Creates an empty body.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a statement to the body.
    pub fn push(&mut self, stmt: Stmt) {
        self.stmts.push(stmt);
    }
}

/// Statements in Loop IR.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Stmt {
    /// An assignment: `%v = op`.
    Assign(ValueId, Op),

    /// A loop construct.
    Loop(Loop),

    /// A conditional branch.
    If(IfStmt),

    /// A store to memory.
    Store(MemRef, Value),

    /// A function call (for external functions).
    Call(Option<ValueId>, Symbol, Vec<Value>),

    /// A return statement.
    Return(Option<Value>),

    /// A barrier for synchronization.
    Barrier(BarrierKind),

    /// A comment/annotation.
    Comment(String),
}

/// A loop construct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Loop {
    /// Unique loop identifier.
    pub id: LoopId,
    /// Loop variable.
    pub var: ValueId,
    /// Lower bound (inclusive).
    pub lower: Value,
    /// Upper bound (exclusive).
    pub upper: Value,
    /// Step size.
    pub step: Value,
    /// Loop body.
    pub body: Body,
    /// Loop attributes.
    pub attrs: LoopAttrs,
}

bitflags! {
    /// Loop attributes for optimization.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LoopAttrs: u32 {
        /// Loop can be parallelized.
        const PARALLEL = 0b0000_0001;
        /// Loop can be vectorized.
        const VECTORIZE = 0b0000_0010;
        /// Loop should be unrolled.
        const UNROLL = 0b0000_0100;
        /// Loop is a reduction loop.
        const REDUCTION = 0b0000_1000;
        /// Loop iterations are independent.
        const INDEPENDENT = 0b0001_0000;
        /// Loop has been tiled.
        const TILED = 0b0010_0000;
        /// Loop is the innermost of a tile.
        const TILE_INNER = 0b0100_0000;
    }
}

/// Loop metadata for optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopMetadata {
    /// Loop identifier.
    pub id: LoopId,
    /// Trip count (iterations).
    pub trip_count: TripCount,
    /// Vectorization width (if applicable).
    pub vector_width: Option<u8>,
    /// Parallel chunk size (if applicable).
    pub parallel_chunk: Option<usize>,
    /// Unroll factor (if applicable).
    pub unroll_factor: Option<u8>,
    /// Dependencies with other loops.
    pub dependencies: Vec<LoopDependency>,
}

/// Trip count information.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TripCount {
    /// Statically known trip count.
    Static(usize),
    /// Dynamic trip count.
    Dynamic,
    /// Bounded trip count (upper bound known).
    Bounded(usize),
}

/// A dependency between loops.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopDependency {
    /// Source loop.
    pub source: LoopId,
    /// Target loop.
    pub target: LoopId,
    /// Dependency type.
    pub kind: DependencyKind,
    /// Distance vector (for affine dependencies).
    pub distance: Option<Vec<i32>>,
}

/// Kinds of dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyKind {
    /// Flow dependency (read after write).
    Flow,
    /// Anti dependency (write after read).
    Anti,
    /// Output dependency (write after write).
    Output,
    /// Input dependency (read after read, for locality).
    Input,
}

/// A conditional statement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IfStmt {
    /// Condition value.
    pub cond: Value,
    /// Then branch.
    pub then_body: Body,
    /// Else branch (optional).
    pub else_body: Option<Body>,
}

/// A value (SSA reference or constant).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// A register/variable reference.
    Var(ValueId, LoopType),
    /// An integer constant.
    IntConst(i64, ScalarType),
    /// A floating-point constant.
    FloatConst(f64, ScalarType),
    /// A boolean constant.
    BoolConst(bool),
    /// Undefined value.
    Undef(LoopType),
}

impl Value {
    /// Returns the type of this value.
    #[must_use]
    pub fn ty(&self) -> LoopType {
        match self {
            Self::Var(_, ty) => ty.clone(),
            Self::IntConst(_, s) => LoopType::Scalar(*s),
            Self::FloatConst(_, s) => LoopType::Scalar(*s),
            Self::BoolConst(_) => LoopType::Scalar(ScalarType::Bool),
            Self::Undef(ty) => ty.clone(),
        }
    }

    /// Creates an integer constant.
    #[must_use]
    pub fn int(n: i64, bits: u8) -> Self {
        Self::IntConst(n, ScalarType::Int(bits))
    }

    /// Creates a 64-bit integer constant.
    #[must_use]
    pub fn i64(n: i64) -> Self {
        Self::int(n, 64)
    }

    /// Creates a float constant.
    #[must_use]
    pub fn float(f: f64, bits: u8) -> Self {
        Self::FloatConst(f, ScalarType::Float(bits))
    }

    /// Creates a 64-bit float constant.
    #[must_use]
    pub fn f64(f: f64) -> Self {
        Self::float(f, 64)
    }
}

/// Operations in Loop IR.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Op {
    /// Load from memory.
    Load(MemRef),

    /// Binary arithmetic operation.
    Binary(BinOp, Value, Value),

    /// Unary operation.
    Unary(UnOp, Value),

    /// Comparison.
    Cmp(CmpOp, Value, Value),

    /// Select (conditional).
    Select(Value, Value, Value),

    /// Cast between types.
    Cast(Value, LoopType),

    /// Vector broadcast (scalar to vector).
    Broadcast(Value, u8),

    /// Vector extract (vector to scalar).
    Extract(Value, u8),

    /// Vector insert.
    Insert(Value, Value, u8),

    /// Vector shuffle.
    Shuffle(Value, Value, Vec<i32>),

    /// Reduction within a vector.
    VecReduce(ReduceOp, Value),

    /// Fused multiply-add: a * b + c.
    Fma(Value, Value, Value),

    /// Pointer arithmetic.
    PtrAdd(Value, Value),

    /// Get pointer to buffer element.
    GetPtr(BufferId, Value),

    /// Phi node (for SSA).
    Phi(Vec<(BlockId, Value)>),
}

/// Binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    // Arithmetic
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Signed division.
    SDiv,
    /// Unsigned division.
    UDiv,
    /// Floating-point division.
    FDiv,
    /// Signed remainder.
    SRem,
    /// Unsigned remainder.
    URem,
    /// Floating-point remainder.
    FRem,

    // Bitwise
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Left shift.
    Shl,
    /// Logical right shift.
    LShr,
    /// Arithmetic right shift.
    AShr,

    // Min/Max
    /// Signed minimum.
    SMin,
    /// Unsigned minimum.
    UMin,
    /// Floating-point minimum.
    FMin,
    /// Signed maximum.
    SMax,
    /// Unsigned maximum.
    UMax,
    /// Floating-point maximum.
    FMax,
}

/// Unary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    /// Negation.
    Neg,
    /// Floating-point negation.
    FNeg,
    /// Bitwise NOT.
    Not,
    /// Absolute value.
    Abs,
    /// Floating-point absolute value.
    FAbs,
    /// Square root.
    Sqrt,
    /// Reciprocal square root.
    Rsqrt,
    /// Floor.
    Floor,
    /// Ceiling.
    Ceil,
    /// Round to nearest.
    Round,
    /// Truncate.
    Trunc,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
}

/// Comparison operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CmpOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Signed less than.
    SLt,
    /// Signed less than or equal.
    SLe,
    /// Signed greater than.
    SGt,
    /// Signed greater than or equal.
    SGe,
    /// Unsigned less than.
    ULt,
    /// Unsigned less than or equal.
    ULe,
    /// Unsigned greater than.
    UGt,
    /// Unsigned greater than or equal.
    UGe,
    /// Floating-point ordered equal.
    OEq,
    /// Floating-point ordered not equal.
    ONe,
    /// Floating-point ordered less than.
    OLt,
    /// Floating-point ordered less than or equal.
    OLe,
    /// Floating-point ordered greater than.
    OGt,
    /// Floating-point ordered greater than or equal.
    OGe,
}

/// Reduction operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum reduction.
    Add,
    /// Product reduction.
    Mul,
    /// Minimum reduction.
    Min,
    /// Maximum reduction.
    Max,
    /// AND reduction.
    And,
    /// OR reduction.
    Or,
    /// XOR reduction.
    Xor,
}

/// A memory reference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemRef {
    /// The buffer being accessed.
    pub buffer: BufferId,
    /// The index/offset.
    pub index: Value,
    /// The element type.
    pub elem_ty: LoopType,
    /// Access pattern information.
    pub access: AccessPattern,
}

/// Memory access patterns for optimization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential access (stride 1).
    Sequential,
    /// Strided access.
    Strided(i64),
    /// Random/indirect access.
    Random,
    /// Broadcast (same element for all iterations).
    Broadcast,
    /// Affine access (linear combination of loop indices).
    Affine(AffineAccess),
}

/// Affine memory access pattern.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineAccess {
    /// Coefficients for each loop variable.
    pub coefficients: SmallVec<[(LoopId, i64); 4]>,
    /// Constant offset.
    pub offset: i64,
}

/// Barrier kinds for synchronization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarrierKind {
    /// Memory fence.
    MemFence,
    /// Full barrier (all threads).
    Full,
    /// Thread group barrier.
    ThreadGroup,
}

/// Errors in Loop IR.
#[derive(Clone, Debug, thiserror::Error, Serialize, Deserialize)]
pub enum LoopIrError {
    /// Type mismatch.
    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch {
        /// Expected type.
        expected: LoopType,
        /// Actual type.
        got: LoopType,
    },

    /// Invalid vector width.
    #[error("invalid vector width {width} for type {ty:?}")]
    InvalidVectorWidth {
        /// The vector width.
        width: u8,
        /// The element type.
        ty: ScalarType,
    },

    /// Out of bounds access.
    #[error("buffer access out of bounds")]
    OutOfBounds,

    /// Invalid loop transformation.
    #[error("invalid loop transformation: {reason}")]
    InvalidTransform {
        /// Reason for the error.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_type_sizes() {
        assert_eq!(ScalarType::Bool.size_bytes(), 1);
        assert_eq!(ScalarType::Int(32).size_bytes(), 4);
        assert_eq!(ScalarType::Float(64).size_bytes(), 8);
    }

    #[test]
    fn test_loop_type_size() {
        assert_eq!(LoopType::Scalar(ScalarType::Float(32)).size_bytes(), 4);
        assert_eq!(LoopType::Vector(ScalarType::Float(32), 8).size_bytes(), 32);
    }

    #[test]
    fn test_value_types() {
        let v = Value::i64(42);
        assert_eq!(v.ty(), LoopType::Scalar(ScalarType::Int(64)));

        let f = Value::f64(3.14);
        assert_eq!(f.ty(), LoopType::Scalar(ScalarType::Float(64)));
    }

    #[test]
    fn test_loop_attrs() {
        let attrs = LoopAttrs::PARALLEL | LoopAttrs::VECTORIZE;
        assert!(attrs.contains(LoopAttrs::PARALLEL));
        assert!(attrs.contains(LoopAttrs::VECTORIZE));
        assert!(!attrs.contains(LoopAttrs::UNROLL));
    }

    #[test]
    fn test_trip_count() {
        let static_trip = TripCount::Static(100);
        assert_eq!(static_trip, TripCount::Static(100));

        let dynamic_trip = TripCount::Dynamic;
        assert_eq!(dynamic_trip, TripCount::Dynamic);
    }
}
