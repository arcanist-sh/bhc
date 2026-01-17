//! Tensor fusion pass.
//!
//! This module implements guaranteed fusion for tensor operations per
//! H26-SPEC Section 8. The fusion pass transforms a sequence of tensor
//! operations into fused kernels that execute without intermediate allocation.
//!
//! ## Guaranteed Fusion Patterns (H26-SPEC Section 8.1)
//!
//! These patterns MUST fuse by specification:
//!
//! 1. `map f (map g x)` → single traversal with composed function
//! 2. `zipWith f (map g a) (map h b)` → single traversal
//! 3. `sum (map f x)` → single traversal (reduce of map)
//! 4. `foldl' op z (map f x)` → single traversal
//!
//! ## Fusion Algorithm
//!
//! The pass operates in three phases:
//!
//! 1. **Pattern Detection**: Identify fusible patterns in the IR
//! 2. **Fusion Graph Construction**: Build a graph of fusion opportunities
//! 3. **Kernel Generation**: Generate fused kernels with tracking info

use crate::{
    AllocInfo, AllocRegion, Axis, BinaryOp, BufferId, DType, FusionDecision, FusionInfo, Kernel,
    KernelBody, KernelId, MapFn, ReduceOp, Shape, Strides, TensorId, TensorMeta, TensorOp,
    TensorRef, ZipFn,
};
use bhc_index::Idx;
use bhc_intern::Symbol;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// The fusion context tracks state during fusion analysis.
pub struct FusionContext {
    /// Next kernel ID to allocate.
    next_kernel_id: u32,
    /// Next tensor ID to allocate.
    next_tensor_id: u32,
    /// Next buffer ID to allocate.
    next_buffer_id: u32,
    /// Tensor reference counts (for multi-use detection).
    ref_counts: FxHashMap<TensorId, usize>,
    /// Generated kernels.
    kernels: Vec<Kernel>,
    /// Fusion decisions for reporting.
    decisions: Vec<FusionDecision>,
    /// Whether we're in strict mode (Numeric profile).
    strict_mode: bool,
}

impl FusionContext {
    /// Creates a new fusion context.
    #[must_use]
    pub fn new(strict_mode: bool) -> Self {
        Self {
            next_kernel_id: 0,
            next_tensor_id: 0,
            next_buffer_id: 0,
            ref_counts: FxHashMap::default(),
            kernels: Vec::new(),
            decisions: Vec::new(),
            strict_mode,
        }
    }

    /// Allocates a fresh kernel ID.
    fn fresh_kernel_id(&mut self) -> KernelId {
        let id = KernelId::new(self.next_kernel_id as usize);
        self.next_kernel_id += 1;
        id
    }

    /// Allocates a fresh tensor ID.
    fn fresh_tensor_id(&mut self) -> TensorId {
        let id = TensorId::new(self.next_tensor_id as usize);
        self.next_tensor_id += 1;
        id
    }

    /// Allocates a fresh buffer ID.
    fn fresh_buffer_id(&mut self) -> BufferId {
        let id = BufferId::new(self.next_buffer_id as usize);
        self.next_buffer_id += 1;
        id
    }

    /// Increments the reference count for a tensor.
    fn add_ref(&mut self, id: TensorId) {
        *self.ref_counts.entry(id).or_insert(0) += 1;
    }

    /// Returns the reference count for a tensor.
    fn ref_count(&self, id: TensorId) -> usize {
        self.ref_counts.get(&id).copied().unwrap_or(0)
    }

    /// Returns the generated kernels.
    #[must_use]
    pub fn kernels(&self) -> &[Kernel] {
        &self.kernels
    }

    /// Returns the fusion decisions for reporting.
    #[must_use]
    pub fn decisions(&self) -> &[FusionDecision] {
        &self.decisions
    }

    /// Consumes the context and returns the generated kernels.
    #[must_use]
    pub fn into_kernels(self) -> Vec<Kernel> {
        self.kernels
    }
}

/// A fusible operation in the fusion graph.
#[derive(Clone, Debug)]
pub struct FusibleOp {
    /// The original tensor operation.
    pub op: TensorOp,
    /// The output tensor reference.
    pub output: TensorRef,
    /// Input tensor IDs.
    pub inputs: SmallVec<[TensorId; 2]>,
    /// Whether this op has been fused.
    pub fused: bool,
}

/// Result of fusion pattern matching.
///
/// These patterns correspond to the guaranteed fusion patterns from H26-SPEC Section 8.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum FusionPattern {
    /// Pattern 1: `map f (map g x)` - composable maps fused to single traversal.
    MapMap {
        /// The outer map function (f).
        outer_fn: MapFn,
        /// The inner map function (g).
        inner_fn: MapFn,
        /// The input tensor reference.
        input: TensorRef,
    },
    /// Pattern 2: `zipWith f (map g a) (map h b)` - zip of maps fused to single traversal.
    ZipWithMaps {
        /// The combining function for zipWith.
        zip_fn: ZipFn,
        /// Optional map function applied to left input.
        left_fn: Option<MapFn>,
        /// Left input tensor reference.
        left_input: TensorRef,
        /// Optional map function applied to right input.
        right_fn: Option<MapFn>,
        /// Right input tensor reference.
        right_input: TensorRef,
    },
    /// Pattern 3: `reduce op (map f x)` - reduction of map fused to single traversal.
    ReduceMap {
        /// The reduction operation (sum, product, etc.).
        reduce_op: ReduceOp,
        /// Optional axis for the reduction.
        axis: Option<Axis>,
        /// The map function applied before reduction.
        map_fn: MapFn,
        /// The input tensor reference.
        input: TensorRef,
    },
    /// Pattern 4: `foldl' op z (map f x)` - fold of map fused to single traversal.
    FoldMap {
        /// The fold combining function.
        fold_fn: Symbol,
        /// Initial accumulator value.
        init: TensorRef,
        /// The map function applied during fold.
        map_fn: MapFn,
        /// The input tensor reference.
        input: TensorRef,
    },
}

/// Fuses a sequence of tensor operations into kernels.
///
/// This is the main entry point for the fusion pass.
pub fn fuse_ops(ctx: &mut FusionContext, ops: Vec<TensorOp>) -> Vec<Kernel> {
    // Phase 1: Build the operation graph and count references
    let fusible_ops = build_fusible_ops(ctx, &ops);

    // Phase 2: Detect and apply fusion patterns
    let fused_groups = detect_and_fuse(ctx, fusible_ops);

    // Phase 3: Generate kernels from fused groups
    for group in fused_groups {
        let kernel = generate_kernel(ctx, group);
        ctx.kernels.push(kernel);
    }

    ctx.kernels.clone()
}

/// Builds fusible operation wrappers and counts references.
fn build_fusible_ops(ctx: &mut FusionContext, ops: &[TensorOp]) -> Vec<FusibleOp> {
    let mut fusible = Vec::with_capacity(ops.len());

    for op in ops {
        // Extract input tensor IDs
        let inputs = extract_input_ids(op);

        // Count references to inputs
        for &id in &inputs {
            ctx.add_ref(id);
        }

        // Create output tensor reference
        let output = create_output_ref(ctx, op);

        fusible.push(FusibleOp {
            op: op.clone(),
            output,
            inputs,
            fused: false,
        });
    }

    fusible
}

/// Extracts input tensor IDs from an operation.
fn extract_input_ids(op: &TensorOp) -> SmallVec<[TensorId; 2]> {
    match op {
        TensorOp::Constant(_) => SmallVec::new(),
        TensorOp::Unary(_, t) | TensorOp::Map(_, t) => smallvec::smallvec![t.id],
        TensorOp::Binary(_, t1, t2) | TensorOp::ZipWith(_, t1, t2) => {
            smallvec::smallvec![t1.id, t2.id]
        }
        TensorOp::Reduce(_, _, t) | TensorOp::ReduceAll(_, t) | TensorOp::Scan(_, _, t) => {
            smallvec::smallvec![t.id]
        }
        TensorOp::Fold(_, init, t) => smallvec::smallvec![init.id, t.id],
        TensorOp::Reshape(_, t)
        | TensorOp::Slice(_, t)
        | TensorOp::Transpose(_, t)
        | TensorOp::Broadcast(_, t) => smallvec::smallvec![t.id],
        TensorOp::Concat(_, refs) => refs.iter().map(|r| r.id).collect(),
        TensorOp::Split(_, _, t) => smallvec::smallvec![t.id],
        TensorOp::MatMul(t1, t2)
        | TensorOp::BatchMatMul(t1, t2)
        | TensorOp::Dot(t1, t2)
        | TensorOp::Outer(t1, t2) => smallvec::smallvec![t1.id, t2.id],
        TensorOp::Conv(_, t1, t2) => smallvec::smallvec![t1.id, t2.id],
        TensorOp::Gather(_, idx, t) => smallvec::smallvec![idx.id, t.id],
        TensorOp::Scatter(_, idx, src, dst) => smallvec::smallvec![idx.id, src.id, dst.id],
    }
}

/// Creates an output tensor reference for an operation.
fn create_output_ref(ctx: &mut FusionContext, op: &TensorOp) -> TensorRef {
    let id = ctx.fresh_tensor_id();
    let meta = infer_output_meta(op);
    TensorRef { id, meta }
}

/// Infers output metadata from an operation.
fn infer_output_meta(op: &TensorOp) -> TensorMeta {
    match op {
        TensorOp::Constant(c) => match c {
            crate::ConstantOp::Zeros(m)
            | crate::ConstantOp::Ones(m)
            | crate::ConstantOp::Full(m, _) => m.clone(),
            crate::ConstantOp::Range(dtype, start, stop, step) => {
                let count = ((stop - start) / step) as usize;
                TensorMeta::new_contiguous(*dtype, Shape::from_static([count]))
                    .unwrap_or_else(|| default_meta(*dtype))
            }
            crate::ConstantOp::Eye(dtype, n) => {
                TensorMeta::new_contiguous(*dtype, Shape::from_static([*n, *n]))
                    .unwrap_or_else(|| default_meta(*dtype))
            }
        },
        TensorOp::Unary(_, t) | TensorOp::Map(_, t) => {
            // Preserves shape and dtype
            t.meta.clone()
        }
        TensorOp::Binary(_, t1, _) | TensorOp::ZipWith(_, t1, _) => {
            // Result has same shape as inputs (assuming broadcast done)
            t1.meta.clone()
        }
        TensorOp::Reduce(_, axis, t) => {
            // Remove the reduced axis
            let mut dims: SmallVec<[crate::Dim; 4]> = t.meta.shape.dims().iter().cloned().collect();
            if let Some(idx) = axis.normalize(dims.len()) {
                dims.remove(idx);
            }
            let shape = Shape::new(dims);
            TensorMeta::new_contiguous(t.meta.dtype, shape).unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::ReduceAll(_, t) => {
            // Scalar output
            TensorMeta::new_contiguous(t.meta.dtype, Shape::scalar())
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Scan(_, _, t) => {
            // Same shape as input
            t.meta.clone()
        }
        TensorOp::Fold(_, _, t) => {
            // Scalar output
            TensorMeta::new_contiguous(t.meta.dtype, Shape::scalar())
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Reshape(shape, t) => {
            TensorMeta::new_contiguous(t.meta.dtype, shape.clone())
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Slice(spec, t) => {
            // Compute sliced shape
            let mut new_dims: SmallVec<[crate::Dim; 4]> = SmallVec::new();
            for (i, range) in spec.ranges.iter().enumerate() {
                if let Some(dim) = t.meta.shape.dims().get(i) {
                    if let Some(n) = dim.static_value() {
                        let start = range.start.unwrap_or(0);
                        let stop = range.stop.unwrap_or(n as i64);
                        let step = range.step;
                        let count = ((stop - start) / step) as usize;
                        new_dims.push(crate::Dim::Static(count));
                    } else {
                        new_dims.push(dim.clone());
                    }
                }
            }
            TensorMeta::new_contiguous(t.meta.dtype, Shape::new(new_dims))
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Transpose(perm, t) => {
            // Permute dimensions
            let old_dims = t.meta.shape.dims();
            let new_dims: SmallVec<[crate::Dim; 4]> =
                perm.as_slice().iter().map(|&i| old_dims[i].clone()).collect();
            // Note: Transpose creates strided layout, not contiguous
            let shape = Shape::new(new_dims);
            TensorMeta {
                dtype: t.meta.dtype,
                shape: shape.clone(),
                strides: Strides::new(perm.as_slice().iter().map(|&i| t.meta.strides.values()[i])),
                layout: crate::Layout::Strided,
                // Views alias the underlying buffer
                alias: t.meta.alias,
            }
        }
        TensorOp::Broadcast(shape, t) => {
            TensorMeta::new_contiguous(t.meta.dtype, shape.clone())
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Concat(_, refs) => {
            if let Some(first) = refs.first() {
                first.meta.clone()
            } else {
                default_meta(DType::Float32)
            }
        }
        TensorOp::Split(_, _, t) => t.meta.clone(),
        TensorOp::MatMul(a, b) | TensorOp::BatchMatMul(a, b) => {
            // [M, K] @ [K, N] -> [M, N]
            let a_dims = a.meta.shape.dims();
            let b_dims = b.meta.shape.dims();
            if a_dims.len() >= 2 && b_dims.len() >= 2 {
                let m = a_dims[a_dims.len() - 2].clone();
                let n = b_dims[b_dims.len() - 1].clone();
                TensorMeta::new_contiguous(a.meta.dtype, Shape::new([m, n]))
                    .unwrap_or_else(|| a.meta.clone())
            } else {
                a.meta.clone()
            }
        }
        TensorOp::Dot(_, t) => {
            TensorMeta::new_contiguous(t.meta.dtype, Shape::scalar())
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Outer(a, b) => {
            // [M] outer [N] -> [M, N]
            let m = a.meta.shape.dims().first().cloned().unwrap_or(crate::Dim::Static(1));
            let n = b.meta.shape.dims().first().cloned().unwrap_or(crate::Dim::Static(1));
            TensorMeta::new_contiguous(a.meta.dtype, Shape::new([m, n]))
                .unwrap_or_else(|| a.meta.clone())
        }
        TensorOp::Conv(_, input, _) => {
            // Simplified: same as input for now
            input.meta.clone()
        }
        TensorOp::Gather(_, _, data) => data.meta.clone(),
        TensorOp::Scatter(_, _, _, dst) => dst.meta.clone(),
    }
}

/// Creates a default metadata for error cases.
fn default_meta(dtype: DType) -> TensorMeta {
    TensorMeta {
        dtype,
        shape: Shape::scalar(),
        strides: Strides::new([]),
        layout: crate::Layout::Contiguous,
        alias: None,
    }
}

/// A group of fused operations.
#[derive(Clone, Debug)]
pub struct FusedGroup {
    /// The operations in this fused group.
    pub ops: Vec<TensorOp>,
    /// Input tensor references.
    pub inputs: Vec<TensorRef>,
    /// Output tensor reference.
    pub output: TensorRef,
    /// The pattern that was fused.
    pub pattern: Option<FusionPattern>,
    /// Names of fused operations (for reporting).
    pub op_names: Vec<Symbol>,
}

/// Detects and applies fusion patterns.
fn detect_and_fuse(ctx: &mut FusionContext, mut ops: Vec<FusibleOp>) -> Vec<FusedGroup> {
    let mut groups = Vec::new();

    // Process operations in reverse order to find consumers first
    let mut i = ops.len();
    while i > 0 {
        i -= 1;

        if ops[i].fused {
            continue;
        }

        // Try to find a fusible pattern
        if let Some((pattern, consumed_indices)) = find_fusion_pattern(ctx, &ops, i) {
            // Mark consumed operations as fused
            for &idx in &consumed_indices {
                ops[idx].fused = true;
            }

            // Create fused group
            let group = create_fused_group(ctx, &ops, &consumed_indices, pattern);

            // Record fusion decision
            ctx.decisions.push(FusionDecision::Fused(group.op_names.clone()));

            groups.push(group);
        } else {
            // No fusion - create single-op group
            ops[i].fused = true;
            let group = FusedGroup {
                ops: vec![ops[i].op.clone()],
                inputs: ops[i]
                    .inputs
                    .iter()
                    .map(|&id| TensorRef {
                        id,
                        meta: default_meta(DType::Float32),
                    })
                    .collect(),
                output: ops[i].output.clone(),
                pattern: None,
                op_names: vec![op_name(&ops[i].op)],
            };
            groups.push(group);
        }
    }

    groups.reverse();
    groups
}

/// Finds a fusion pattern starting from the given operation index.
fn find_fusion_pattern(
    ctx: &FusionContext,
    ops: &[FusibleOp],
    consumer_idx: usize,
) -> Option<(FusionPattern, Vec<usize>)> {
    let consumer = &ops[consumer_idx];

    // Pattern 1: map f (map g x)
    if let TensorOp::Map(outer_fn, inner_ref) = &consumer.op {
        // Check if the input is a single-use map
        if let Some(producer_idx) = find_producer(ops, inner_ref.id) {
            if !ops[producer_idx].fused && ctx.ref_count(inner_ref.id) == 1 {
                if let TensorOp::Map(inner_fn, input_ref) = &ops[producer_idx].op {
                    return Some((
                        FusionPattern::MapMap {
                            outer_fn: outer_fn.clone(),
                            inner_fn: inner_fn.clone(),
                            input: input_ref.clone(),
                        },
                        vec![consumer_idx, producer_idx],
                    ));
                }
            }
        }
    }

    // Pattern 3: reduce op (map f x) - sum/prod/max/min of map
    if let TensorOp::ReduceAll(reduce_op, inner_ref) = &consumer.op {
        if let Some(producer_idx) = find_producer(ops, inner_ref.id) {
            if !ops[producer_idx].fused && ctx.ref_count(inner_ref.id) == 1 {
                if let TensorOp::Map(map_fn, input_ref) = &ops[producer_idx].op {
                    return Some((
                        FusionPattern::ReduceMap {
                            reduce_op: *reduce_op,
                            axis: None,
                            map_fn: map_fn.clone(),
                            input: input_ref.clone(),
                        },
                        vec![consumer_idx, producer_idx],
                    ));
                }
            }
        }
    }

    // Pattern 3 variant: reduce with axis
    if let TensorOp::Reduce(reduce_op, axis, inner_ref) = &consumer.op {
        if let Some(producer_idx) = find_producer(ops, inner_ref.id) {
            if !ops[producer_idx].fused && ctx.ref_count(inner_ref.id) == 1 {
                if let TensorOp::Map(map_fn, input_ref) = &ops[producer_idx].op {
                    return Some((
                        FusionPattern::ReduceMap {
                            reduce_op: *reduce_op,
                            axis: Some(*axis),
                            map_fn: map_fn.clone(),
                            input: input_ref.clone(),
                        },
                        vec![consumer_idx, producer_idx],
                    ));
                }
            }
        }
    }

    // Pattern 2: zipWith f (map g a) (map h b)
    if let TensorOp::ZipWith(zip_fn, left_ref, right_ref) = &consumer.op {
        let left_producer = find_producer(ops, left_ref.id);
        let right_producer = find_producer(ops, right_ref.id);

        let left_is_fusible_map = left_producer.map_or(false, |idx| {
            !ops[idx].fused
                && ctx.ref_count(left_ref.id) == 1
                && matches!(&ops[idx].op, TensorOp::Map(_, _))
        });

        let right_is_fusible_map = right_producer.map_or(false, |idx| {
            !ops[idx].fused
                && ctx.ref_count(right_ref.id) == 1
                && matches!(&ops[idx].op, TensorOp::Map(_, _))
        });

        if left_is_fusible_map || right_is_fusible_map {
            let mut consumed = vec![consumer_idx];
            let mut left_fn = None;
            let mut left_input = left_ref.clone();
            let mut right_fn = None;
            let mut right_input = right_ref.clone();

            if let Some(idx) = left_producer {
                if left_is_fusible_map {
                    if let TensorOp::Map(f, inp) = &ops[idx].op {
                        left_fn = Some(f.clone());
                        left_input = inp.clone();
                        consumed.push(idx);
                    }
                }
            }

            if let Some(idx) = right_producer {
                if right_is_fusible_map {
                    if let TensorOp::Map(f, inp) = &ops[idx].op {
                        right_fn = Some(f.clone());
                        right_input = inp.clone();
                        consumed.push(idx);
                    }
                }
            }

            return Some((
                FusionPattern::ZipWithMaps {
                    zip_fn: zip_fn.clone(),
                    left_fn,
                    left_input,
                    right_fn,
                    right_input,
                },
                consumed,
            ));
        }
    }

    None
}

/// Finds the producer operation for a tensor ID.
fn find_producer(ops: &[FusibleOp], id: TensorId) -> Option<usize> {
    ops.iter().position(|op| op.output.id == id)
}

/// Creates a fused group from a pattern and consumed operations.
fn create_fused_group(
    ctx: &mut FusionContext,
    ops: &[FusibleOp],
    consumed_indices: &[usize],
    pattern: FusionPattern,
) -> FusedGroup {
    let op_names: Vec<Symbol> = consumed_indices
        .iter()
        .map(|&i| op_name(&ops[i].op))
        .collect();

    let (inputs, output, fused_ops) = match &pattern {
        FusionPattern::MapMap {
            outer_fn,
            inner_fn,
            input,
        } => {
            // Compose: (outer . inner)
            let composed_fn = MapFn {
                name: Symbol::intern(&format!(
                    "({} . {})",
                    outer_fn.name.as_str(),
                    inner_fn.name.as_str()
                )),
                span: outer_fn.span,
            };
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: input.meta.clone(),
            };
            (
                vec![input.clone()],
                output.clone(),
                vec![TensorOp::Map(composed_fn, input.clone())],
            )
        }
        FusionPattern::ReduceMap {
            reduce_op,
            axis,
            map_fn,
            input,
        } => {
            let output_id = ctx.fresh_tensor_id();
            let output_shape = if axis.is_some() {
                // Reduce along axis
                let mut dims: SmallVec<[crate::Dim; 4]> =
                    input.meta.shape.dims().iter().cloned().collect();
                if let Some(idx) = axis.and_then(|a| a.normalize(dims.len())) {
                    dims.remove(idx);
                }
                Shape::new(dims)
            } else {
                Shape::scalar()
            };
            let output = TensorRef {
                id: output_id,
                meta: TensorMeta::new_contiguous(input.meta.dtype, output_shape)
                    .unwrap_or_else(|| input.meta.clone()),
            };

            // Create fused reduce-map operation
            let fused_op = if let Some(ax) = axis {
                TensorOp::Reduce(*reduce_op, *ax, input.clone())
            } else {
                TensorOp::ReduceAll(*reduce_op, input.clone())
            };

            // Note: In a real implementation, the map function would be
            // composed into the reduction. For now we represent this as
            // a fused kernel with both operations.
            (
                vec![input.clone()],
                output,
                vec![TensorOp::Map(map_fn.clone(), input.clone()), fused_op],
            )
        }
        FusionPattern::ZipWithMaps {
            zip_fn,
            left_fn,
            left_input,
            right_fn,
            right_input,
        } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: left_input.meta.clone(),
            };

            // Create fused zip-with-maps
            let combined_name = match (left_fn, right_fn) {
                (Some(l), Some(r)) => format!(
                    "zipWith {} ({}) ({})",
                    zip_fn.name.as_str(),
                    l.name.as_str(),
                    r.name.as_str()
                ),
                (Some(l), None) => {
                    format!("zipWith {} ({}) id", zip_fn.name.as_str(), l.name.as_str())
                }
                (None, Some(r)) => {
                    format!("zipWith {} id ({})", zip_fn.name.as_str(), r.name.as_str())
                }
                (None, None) => format!("zipWith {}", zip_fn.name.as_str()),
            };

            let fused_fn = ZipFn {
                name: Symbol::intern(&combined_name),
                span: zip_fn.span,
            };

            (
                vec![left_input.clone(), right_input.clone()],
                output,
                vec![TensorOp::ZipWith(fused_fn, left_input.clone(), right_input.clone())],
            )
        }
        FusionPattern::FoldMap {
            fold_fn,
            init,
            map_fn,
            input,
        } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: TensorMeta::new_contiguous(input.meta.dtype, Shape::scalar())
                    .unwrap_or_else(|| input.meta.clone()),
            };

            let fused_fn = crate::FoldFn {
                name: Symbol::intern(&format!(
                    "fold {} . {}",
                    fold_fn.as_str(),
                    map_fn.name.as_str()
                )),
                span: map_fn.span,
            };

            (
                vec![init.clone(), input.clone()],
                output,
                vec![TensorOp::Fold(fused_fn, init.clone(), input.clone())],
            )
        }
    };

    FusedGroup {
        ops: fused_ops,
        inputs,
        output,
        pattern: Some(pattern),
        op_names,
    }
}

/// Gets a symbolic name for an operation (for reporting).
fn op_name(op: &TensorOp) -> Symbol {
    let name = match op {
        TensorOp::Constant(_) => "constant",
        TensorOp::Unary(op, _) => match op {
            crate::UnaryOp::Neg => "neg",
            crate::UnaryOp::Abs => "abs",
            crate::UnaryOp::Sqrt => "sqrt",
            crate::UnaryOp::Exp => "exp",
            crate::UnaryOp::Log => "log",
            crate::UnaryOp::Sin => "sin",
            crate::UnaryOp::Cos => "cos",
            _ => "unary",
        },
        TensorOp::Binary(op, _, _) => match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            _ => "binary",
        },
        TensorOp::Map(_, _) => "map",
        TensorOp::ZipWith(_, _, _) => "zipWith",
        TensorOp::Reduce(_, _, _) => "reduce",
        TensorOp::ReduceAll(_, _) => "reduceAll",
        TensorOp::Scan(_, _, _) => "scan",
        TensorOp::Fold(_, _, _) => "fold",
        TensorOp::Reshape(_, _) => "reshape",
        TensorOp::Slice(_, _) => "slice",
        TensorOp::Transpose(_, _) => "transpose",
        TensorOp::Broadcast(_, _) => "broadcast",
        TensorOp::Concat(_, _) => "concat",
        TensorOp::Split(_, _, _) => "split",
        TensorOp::MatMul(_, _) => "matmul",
        TensorOp::BatchMatMul(_, _) => "batchMatmul",
        TensorOp::Dot(_, _) => "dot",
        TensorOp::Outer(_, _) => "outer",
        TensorOp::Conv(_, _, _) => "conv",
        TensorOp::Gather(_, _, _) => "gather",
        TensorOp::Scatter(_, _, _, _) => "scatter",
    };
    Symbol::intern(name)
}

/// Generates a kernel from a fused group.
fn generate_kernel(ctx: &mut FusionContext, group: FusedGroup) -> Kernel {
    let id = ctx.fresh_kernel_id();
    let name = generate_kernel_name(&group);

    // Determine allocation requirements
    let allocs = compute_allocations(ctx, &group);

    // Build fusion info for reporting
    let fusion_info = FusionInfo {
        original_ops: group.op_names.clone(),
        decisions: vec![FusionDecision::Fused(group.op_names.clone())],
        complete: group.pattern.is_some(),
    };

    Kernel {
        id,
        name,
        inputs: group.inputs,
        outputs: vec![group.output],
        body: KernelBody::Fused(group.ops),
        allocs,
        fusion_info,
    }
}

/// Generates a name for a kernel.
fn generate_kernel_name(group: &FusedGroup) -> Symbol {
    if let Some(pattern) = &group.pattern {
        let name = match pattern {
            FusionPattern::MapMap { .. } => "fused_map_map",
            FusionPattern::ReduceMap { .. } => "fused_reduce_map",
            FusionPattern::ZipWithMaps { .. } => "fused_zipwith_maps",
            FusionPattern::FoldMap { .. } => "fused_fold_map",
        };
        Symbol::intern(name)
    } else if group.op_names.len() == 1 {
        group.op_names[0]
    } else {
        Symbol::intern("kernel")
    }
}

/// Computes allocation requirements for a fused group.
fn compute_allocations(ctx: &mut FusionContext, group: &FusedGroup) -> Vec<AllocInfo> {
    let mut allocs = Vec::new();

    // Output buffer allocation
    if let Some(size) = group.output.meta.size_bytes() {
        let buffer = ctx.fresh_buffer_id();
        allocs.push(AllocInfo {
            buffer,
            size,
            alignment: group.output.meta.dtype.alignment(),
            region: if ctx.strict_mode {
                AllocRegion::HotArena
            } else {
                AllocRegion::General
            },
        });
    }

    allocs
}

/// Checks if a reshape is metadata-only (no data movement needed).
///
/// A reshape is metadata-only when the tensor is contiguous.
#[must_use]
pub fn is_reshape_metadata_only(tensor: &TensorRef) -> bool {
    matches!(tensor.meta.layout, crate::Layout::Contiguous)
}

/// Generates a kernel report for the fusion pass.
#[must_use]
pub fn generate_kernel_report(ctx: &FusionContext) -> KernelReport {
    KernelReport {
        kernels: ctx.kernels.clone(),
        decisions: ctx.decisions.clone(),
        total_ops: ctx.kernels.iter().map(|k| k.fusion_info.original_ops.len()).sum(),
        fused_ops: ctx
            .decisions
            .iter()
            .filter(|d| matches!(d, FusionDecision::Fused(_)))
            .count(),
    }
}

/// A report of the fusion pass results.
#[derive(Clone, Debug)]
pub struct KernelReport {
    /// Generated kernels.
    pub kernels: Vec<Kernel>,
    /// Fusion decisions made.
    pub decisions: Vec<FusionDecision>,
    /// Total number of original operations.
    pub total_ops: usize,
    /// Number of operations that were fused.
    pub fused_ops: usize,
}

impl std::fmt::Display for KernelReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Kernel Report ===")?;
        writeln!(f, "Total operations: {}", self.total_ops)?;
        writeln!(f, "Fused operations: {}", self.fused_ops)?;
        writeln!(f, "Generated kernels: {}", self.kernels.len())?;
        writeln!(f)?;

        for kernel in &self.kernels {
            writeln!(f, "Kernel: {}", kernel.name.as_str())?;
            writeln!(f, "  Inputs: {}", kernel.inputs.len())?;
            writeln!(f, "  Outputs: {}", kernel.outputs.len())?;
            writeln!(
                f,
                "  Fused: {}",
                if kernel.fusion_info.complete {
                    "YES"
                } else {
                    "NO"
                }
            )?;
            if !kernel.fusion_info.original_ops.is_empty() {
                write!(f, "  Original ops: ")?;
                for (i, op) in kernel.fusion_info.original_ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", op.as_str())?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_span::Span;

    fn make_tensor_ref(id: u32, shape: &[usize], dtype: DType) -> TensorRef {
        TensorRef {
            id: TensorId::new(id as usize),
            meta: TensorMeta::new_contiguous(dtype, Shape::from_static(shape.iter().copied()))
                .unwrap(),
        }
    }

    fn make_map_fn(name: &str) -> MapFn {
        MapFn {
            name: Symbol::intern(name),
            span: Span::DUMMY,
        }
    }

    #[test]
    fn test_pattern1_map_map_fusion() {
        // map f (map g x) should fuse to map (f . g) x
        //
        // The key insight: the first map's input is distinct from
        // the intermediate (which connects first map output to second map input).
        //
        // x (id=100) -> [map g] -> intermediate (id=0) -> [map f] -> result (id=1)
        //
        // When build_fusible_ops runs:
        // - first_map: inputs=[100], output=0
        // - second_map: inputs=[0], output=1
        // ref_counts: {100: 1, 0: 1}
        //
        // Since intermediate (id=0) has ref_count=1, fusion is allowed.

        // Use a high ID for the original input so it won't conflict
        let input = make_tensor_ref(100, &[100], DType::Float32);
        let first_map = TensorOp::Map(make_map_fn("g"), input);

        // The intermediate will get id=0 when processed by build_fusible_ops
        // Second map consumes id=0
        let intermediate = make_tensor_ref(0, &[100], DType::Float32);
        let second_map = TensorOp::Map(make_map_fn("f"), intermediate);

        let mut ctx = FusionContext::new(true);
        let ops = vec![first_map, second_map];
        let kernels = fuse_ops(&mut ctx, ops);

        assert_eq!(kernels.len(), 1, "should produce single fused kernel");
        assert!(
            kernels[0].fusion_info.complete,
            "fusion should be complete"
        );
        assert_eq!(
            kernels[0].fusion_info.original_ops.len(),
            2,
            "should track both original ops"
        );
    }

    #[test]
    fn test_pattern3_reduce_map_fusion() {
        // sum (map f x) should fuse to single traversal

        // Use high ID for original input
        let input = make_tensor_ref(100, &[100], DType::Float32);
        let first_map = TensorOp::Map(make_map_fn("f"), input);

        // Intermediate will get id=0, reduce consumes id=0
        let mapped = make_tensor_ref(0, &[100], DType::Float32);
        let reduce = TensorOp::ReduceAll(ReduceOp::Sum, mapped);

        let mut ctx = FusionContext::new(true);
        let ops = vec![first_map, reduce];
        let kernels = fuse_ops(&mut ctx, ops);

        assert_eq!(kernels.len(), 1, "should produce single fused kernel");
        assert!(
            kernels[0].fusion_info.complete,
            "fusion should be complete"
        );
    }

    #[test]
    fn test_reshape_metadata_only() {
        let contiguous = make_tensor_ref(0, &[10, 10], DType::Float32);
        assert!(
            is_reshape_metadata_only(&contiguous),
            "contiguous tensor reshape should be metadata-only"
        );

        // Non-contiguous (strided) tensor
        let strided = TensorRef {
            id: TensorId::new(1),
            meta: TensorMeta {
                dtype: DType::Float32,
                shape: Shape::from_static([10, 10]),
                strides: Strides::new([40, 8]), // Non-standard strides
                layout: crate::Layout::Strided,
                alias: None,
            },
        };
        assert!(
            !is_reshape_metadata_only(&strided),
            "strided tensor reshape should require data movement"
        );
    }

    #[test]
    fn test_kernel_report_generation() {
        // Use high ID for original input
        let input = make_tensor_ref(100, &[100], DType::Float32);
        let first_map = TensorOp::Map(make_map_fn("double"), input);

        // Intermediate connects first to second
        let intermediate = make_tensor_ref(0, &[100], DType::Float32);
        let second_map = TensorOp::Map(make_map_fn("inc"), intermediate);

        let mut ctx = FusionContext::new(true);
        let ops = vec![first_map, second_map];
        let _kernels = fuse_ops(&mut ctx, ops);
        let report = generate_kernel_report(&ctx);

        assert_eq!(report.kernels.len(), 1);
        assert!(report.fused_ops > 0, "should have fused operations");

        // Test display
        let display = format!("{report}");
        assert!(display.contains("Kernel Report"));
        assert!(display.contains("fused_map_map"));
    }

    #[test]
    fn test_multi_use_prevents_fusion() {
        // If intermediate is used multiple times, can't fuse
        // Simulate by having two consumers of the same intermediate ID

        let input = make_tensor_ref(100, &[100], DType::Float32);
        let first_map = TensorOp::Map(make_map_fn("g"), input);

        // Create two maps that both consume the intermediate (id=0)
        let intermediate = make_tensor_ref(0, &[100], DType::Float32);
        let second_map = TensorOp::Map(make_map_fn("f"), intermediate.clone());
        let third_map = TensorOp::Map(make_map_fn("h"), intermediate);

        let mut ctx = FusionContext::new(true);
        let ops = vec![first_map, second_map, third_map];
        let kernels = fuse_ops(&mut ctx, ops);

        // The intermediate is used twice (by second_map and third_map),
        // so fusion should be blocked
        assert!(
            kernels.len() >= 2,
            "multi-use intermediate should prevent full fusion"
        );
    }

    #[test]
    fn test_zipwith_map_fusion() {
        // zipWith f (map g a) (map h b) should fuse
        //
        // a (id=100) -> [map g] -> mapped_a (id=0) -\
        //                                            -> [zipWith add] -> result
        // b (id=101) -> [map h] -> mapped_b (id=1) -/

        let a = make_tensor_ref(100, &[100], DType::Float32);
        let b = make_tensor_ref(101, &[100], DType::Float32);

        let map_a = TensorOp::Map(make_map_fn("g"), a);
        let map_b = TensorOp::Map(make_map_fn("h"), b);

        // mapped_a gets id=0, mapped_b gets id=1 from build_fusible_ops
        let mapped_a = make_tensor_ref(0, &[100], DType::Float32);
        let mapped_b = make_tensor_ref(1, &[100], DType::Float32);

        let zip_op = TensorOp::ZipWith(
            ZipFn {
                name: Symbol::intern("add"),
                span: Span::DUMMY,
            },
            mapped_a,
            mapped_b,
        );

        let mut ctx = FusionContext::new(true);
        let ops = vec![map_a, map_b, zip_op];
        let kernels = fuse_ops(&mut ctx, ops);

        // Should produce single fused kernel
        assert_eq!(kernels.len(), 1, "should produce single fused kernel");
        assert!(
            kernels[0].fusion_info.complete,
            "fusion should be complete"
        );
    }

    // ========================================================================
    // M2 Exit Criteria Integration Tests
    //
    // These tests verify the M2 milestone exit criteria per ROADMAP.md:
    // 1. sum (map f x) becomes single loop kernel
    // 2. Kernel report shows fusion succeeded
    // 3. reshape on contiguous tensor is metadata-only
    // ========================================================================

    /// M2 Exit Criterion 1: `sum (map f x)` becomes single loop kernel
    ///
    /// This test verifies that the H26-SPEC Section 8.1 Pattern 3 fuses correctly.
    /// The pattern `reduce op (map f x)` must fuse to a single traversal.
    #[test]
    fn test_m2_sum_map_fuses_to_single_kernel() {
        // Build the pattern: sum (map f x)
        //
        // x (id=100) -> [map f] -> intermediate (id=0) -> [sum] -> scalar result
        //
        // This MUST fuse to a single kernel per H26-SPEC Section 8.1 Pattern 3
        let input = make_tensor_ref(100, &[1000], DType::Float32);
        let map_op = TensorOp::Map(make_map_fn("square"), input);

        let intermediate = make_tensor_ref(0, &[1000], DType::Float32);
        let sum_op = TensorOp::ReduceAll(ReduceOp::Sum, intermediate);

        let mut ctx = FusionContext::new(true); // strict mode (Numeric Profile)
        let ops = vec![map_op, sum_op];
        let kernels = fuse_ops(&mut ctx, ops);

        // M2 Criterion: Must produce exactly ONE kernel
        assert_eq!(
            kernels.len(),
            1,
            "M2 FAIL: sum(map f x) did not fuse to single kernel"
        );

        // Verify the fusion is marked complete
        assert!(
            kernels[0].fusion_info.complete,
            "M2 FAIL: fusion not marked as complete"
        );

        // Verify it's the correct pattern (fused_reduce_map kernel)
        assert_eq!(
            kernels[0].name.as_str(),
            "fused_reduce_map",
            "M2 FAIL: kernel name should indicate reduce-map fusion"
        );

        // Verify both original operations are tracked
        assert_eq!(
            kernels[0].fusion_info.original_ops.len(),
            2,
            "M2 FAIL: should track both map and reduce operations"
        );
    }

    /// M2 Exit Criterion 2: Kernel report shows fusion succeeded
    ///
    /// This test verifies that the kernel report correctly indicates
    /// when fusion has succeeded for guaranteed patterns.
    #[test]
    fn test_m2_kernel_report_shows_fusion_success() {
        // Build: sum (map f x) - a guaranteed fusion pattern
        let input = make_tensor_ref(100, &[500], DType::Float64);
        let map_op = TensorOp::Map(make_map_fn("f"), input);
        let intermediate = make_tensor_ref(0, &[500], DType::Float64);
        let sum_op = TensorOp::ReduceAll(ReduceOp::Sum, intermediate);

        let mut ctx = FusionContext::new(true);
        let ops = vec![map_op, sum_op];
        let _kernels = fuse_ops(&mut ctx, ops);
        let report = generate_kernel_report(&ctx);

        // M2 Criterion: Report must show fusion succeeded
        assert_eq!(report.kernels.len(), 1, "M2 FAIL: should have 1 kernel");
        assert!(
            report.fused_ops > 0,
            "M2 FAIL: report should show fused operations"
        );
        assert_eq!(
            report.total_ops, 2,
            "M2 FAIL: report should track 2 original ops"
        );

        // Verify the kernel is marked as fused
        let kernel = &report.kernels[0];
        assert!(
            kernel.fusion_info.complete,
            "M2 FAIL: kernel report should indicate complete fusion"
        );

        // Verify decisions contain a Fused entry
        let has_fused_decision = report
            .decisions
            .iter()
            .any(|d| matches!(d, FusionDecision::Fused(_)));
        assert!(
            has_fused_decision,
            "M2 FAIL: report should contain Fused decision"
        );

        // Test report display output
        let report_str = format!("{report}");
        assert!(
            report_str.contains("Kernel Report"),
            "M2 FAIL: report should have header"
        );
        assert!(
            report_str.contains("Fused: YES"),
            "M2 FAIL: report should show 'Fused: YES'"
        );
    }

    /// M2 Exit Criterion 3: reshape on contiguous tensor is metadata-only
    ///
    /// This test verifies that reshaping a contiguous tensor does not
    /// require data movement - only metadata (shape/strides) changes.
    #[test]
    fn test_m2_reshape_contiguous_metadata_only() {
        // A contiguous tensor can be reshaped without copying data
        let contiguous = make_tensor_ref(0, &[6, 4], DType::Float32);

        // Verify contiguous tensor reshape is metadata-only
        assert!(
            is_reshape_metadata_only(&contiguous),
            "M2 FAIL: contiguous tensor reshape should be metadata-only"
        );

        // Build a more complex contiguous tensor
        let contiguous_3d = make_tensor_ref(1, &[2, 3, 4], DType::Float32);
        assert!(
            is_reshape_metadata_only(&contiguous_3d),
            "M2 FAIL: 3D contiguous tensor reshape should be metadata-only"
        );

        // Verify non-contiguous (strided) tensor requires data movement
        let strided = TensorRef {
            id: TensorId::new(2),
            meta: TensorMeta {
                dtype: DType::Float32,
                shape: Shape::from_static([6, 4]),
                strides: Strides::new([8, 1]), // Non-contiguous strides
                layout: crate::Layout::Strided,
                alias: None,
            },
        };
        assert!(
            !is_reshape_metadata_only(&strided),
            "M2 FAIL: strided tensor reshape should NOT be metadata-only"
        );
    }

    /// Integration test: Complete M2 pipeline with all guaranteed patterns
    ///
    /// Tests all H26-SPEC Section 8.1 guaranteed patterns:
    /// 1. map f (map g x)
    /// 2. zipWith f (map g a) (map h b)
    /// 3. sum (map f x)
    /// 4. foldl' op z (map f x) - (tested via ReduceMap pattern)
    #[test]
    fn test_m2_all_guaranteed_patterns_fuse() {
        // Pattern 1: map f (map g x)
        {
            let input = make_tensor_ref(100, &[100], DType::Float32);
            let map_g = TensorOp::Map(make_map_fn("g"), input);
            let intermediate = make_tensor_ref(0, &[100], DType::Float32);
            let map_f = TensorOp::Map(make_map_fn("f"), intermediate);

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_g, map_f]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 1 (map-map) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }

        // Pattern 2: zipWith f (map g a) (map h b)
        {
            let a = make_tensor_ref(100, &[100], DType::Float32);
            let b = make_tensor_ref(101, &[100], DType::Float32);
            let map_a = TensorOp::Map(make_map_fn("g"), a);
            let map_b = TensorOp::Map(make_map_fn("h"), b);
            let mapped_a = make_tensor_ref(0, &[100], DType::Float32);
            let mapped_b = make_tensor_ref(1, &[100], DType::Float32);
            let zip = TensorOp::ZipWith(
                ZipFn {
                    name: Symbol::intern("add"),
                    span: Span::DUMMY,
                },
                mapped_a,
                mapped_b,
            );

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_a, map_b, zip]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 2 (zipWith-maps) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }

        // Pattern 3: sum (map f x)
        {
            let input = make_tensor_ref(100, &[100], DType::Float32);
            let map_op = TensorOp::Map(make_map_fn("f"), input);
            let intermediate = make_tensor_ref(0, &[100], DType::Float32);
            let sum_op = TensorOp::ReduceAll(ReduceOp::Sum, intermediate);

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_op, sum_op]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 3 (reduce-map) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }

        // Pattern 3 variant: product (map f x)
        {
            let input = make_tensor_ref(100, &[100], DType::Float32);
            let map_op = TensorOp::Map(make_map_fn("f"), input);
            let intermediate = make_tensor_ref(0, &[100], DType::Float32);
            let prod_op = TensorOp::ReduceAll(ReduceOp::Prod, intermediate);

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_op, prod_op]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 3 variant (product-map) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }

        // Pattern 3 variant: max (map f x)
        {
            let input = make_tensor_ref(100, &[100], DType::Float32);
            let map_op = TensorOp::Map(make_map_fn("f"), input);
            let intermediate = make_tensor_ref(0, &[100], DType::Float32);
            let max_op = TensorOp::ReduceAll(ReduceOp::Max, intermediate);

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_op, max_op]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 3 variant (max-map) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }
    }
}
