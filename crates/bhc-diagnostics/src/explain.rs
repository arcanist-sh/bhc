//! Error code explanations.
//!
//! This module provides detailed explanations for error codes, accessible
//! via the `--explain` flag (e.g., `bhc --explain E0001`).
//!
//! ## Error Code Conventions
//!
//! - `E0xxx`: Type errors
//! - `E00xx`: Basic type mismatches (E0001-E0019)
//! - `E002x`: Shape/dimension errors (E0020-E0039)
//! - `E003x`: Tensor operation errors (E0030-E0039)
//! - `E004x`: Pattern matching errors (E0040-E0049)
//! - `E005x`: Module/import errors (E0050-E0059)
//! - `W0xxx`: Warnings
//! - `W001x`: Unused bindings (W0010-W0019)
//! - `W002x`: Deprecated features (W0020-W0029)

use std::collections::HashMap;
use std::sync::LazyLock;

/// An error code explanation.
#[derive(Clone, Debug)]
pub struct ErrorExplanation {
    /// The error code.
    pub code: &'static str,
    /// A brief title for the error.
    pub title: &'static str,
    /// A detailed explanation.
    pub explanation: &'static str,
    /// Example code that triggers this error.
    pub example: Option<&'static str>,
    /// Example of the correct code.
    pub correct_example: Option<&'static str>,
}

/// Registry of all error code explanations.
static ERROR_REGISTRY: LazyLock<HashMap<&'static str, ErrorExplanation>> = LazyLock::new(|| {
    let mut map = HashMap::new();

    // === Type Errors (E0001-E0019) ===

    map.insert(
        "E0001",
        ErrorExplanation {
            code: "E0001",
            title: "Type mismatch",
            explanation: r#"
This error occurs when the type checker expects one type but finds another.

Type mismatches commonly occur when:
- A function is called with an argument of the wrong type
- A variable is used in a context that requires a different type
- A return value doesn't match the function's declared return type

The compiler shows the expected type and the actual type found. Check that
your types align, or add explicit type conversions where needed.
"#,
            example: Some(
                r#"
foo :: Int -> Int
foo x = "hello"  -- Error: expected Int, found String
"#,
            ),
            correct_example: Some(
                r#"
foo :: Int -> Int
foo x = x + 1
"#,
            ),
        },
    );

    map.insert(
        "E0002",
        ErrorExplanation {
            code: "E0002",
            title: "Infinite type (occurs check)",
            explanation: r#"
This error occurs when type inference would create an infinite type,
typically when a value is used in a way that would require it to contain
itself.

This is detected by the "occurs check" during unification. If a type
variable would need to be unified with a type containing that same
variable, it would create an infinite loop in the type.

Common causes:
- Recursive data without proper type annotations
- Accidentally creating circular references in types
"#,
            example: Some(
                r#"
-- This creates an infinite type: a = [a]
foo x = [x, foo x]
"#,
            ),
            correct_example: Some(
                r#"
-- Use a recursive data type instead
data Tree a = Leaf a | Node [Tree a]

foo :: a -> Tree a
foo x = Node [Leaf x, foo x]
"#,
            ),
        },
    );

    map.insert(
        "E0003",
        ErrorExplanation {
            code: "E0003",
            title: "Unbound variable",
            explanation: r#"
This error occurs when you use a variable name that hasn't been defined
in the current scope.

Common causes:
- Typo in the variable name
- Using a variable before it's defined
- Variable defined in a different scope (e.g., inside a let or lambda)
- Forgot to import a module
"#,
            example: Some(
                r#"
foo = x + 1  -- Error: x is not defined
"#,
            ),
            correct_example: Some(
                r#"
foo x = x + 1  -- x is now a parameter
"#,
            ),
        },
    );

    // === Shape Errors (E0020-E0029) ===

    map.insert(
        "E0020",
        ErrorExplanation {
            code: "E0020",
            title: "Dimension mismatch",
            explanation: r#"
This error occurs when tensor dimensions don't match as required by an
operation.

For example, matrix multiplication requires the inner dimensions to match:
  matmul :: Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a

The 'k' dimension (columns of first matrix, rows of second) must be equal.

Common causes:
- Matrices with incompatible shapes for multiplication
- Elementwise operations on tensors of different shapes
- Incorrect reshape operations
"#,
            example: Some(
                r#"
-- Shapes: [3, 5] × [7, 4] - inner dimensions 5 ≠ 7
result = matmul a b  -- Error: dimension mismatch
"#,
            ),
            correct_example: Some(
                r#"
-- Shapes: [3, 5] × [5, 4] - inner dimensions match
result = matmul a b  -- OK: produces [3, 4]
"#,
            ),
        },
    );

    map.insert(
        "E0023",
        ErrorExplanation {
            code: "E0023",
            title: "Shape rank mismatch",
            explanation: r#"
This error occurs when a tensor has a different number of dimensions
(rank) than expected.

For example:
- A function expecting a matrix (rank 2) receives a vector (rank 1)
- A function expecting a vector (rank 1) receives a scalar (rank 0)

Check that your tensors have the correct number of dimensions.
"#,
            example: Some(
                r#"
-- matmul expects rank-2 tensors (matrices)
a :: Tensor '[10] Float      -- rank 1 (vector)
b :: Tensor '[10, 5] Float   -- rank 2 (matrix)
result = matmul a b  -- Error: rank mismatch
"#,
            ),
            correct_example: Some(
                r#"
-- Both operands are rank-2
a :: Tensor '[1, 10] Float   -- rank 2 (row vector as matrix)
b :: Tensor '[10, 5] Float   -- rank 2 (matrix)
result = matmul a b  -- OK: produces [1, 5]
"#,
            ),
        },
    );

    // === Tensor Operation Errors (E0030-E0039) ===

    map.insert(
        "E0030",
        ErrorExplanation {
            code: "E0030",
            title: "Matrix multiplication dimension mismatch",
            explanation: r#"
This error occurs specifically during matrix multiplication when the inner
dimensions don't match.

Matrix multiplication has the signature:
  matmul :: Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a

The second dimension of the first matrix (k, the number of columns) must
equal the first dimension of the second matrix (k, the number of rows).

Visual representation:
  [m × k] @ [k × n] = [m × n]
       └───┴──── these must match

Common fixes:
- Transpose one of the matrices
- Swap the order of arguments
- Reshape to get compatible dimensions
"#,
            example: Some(
                r#"
weights :: Tensor '[768, 512] Float
input   :: Tensor '[1024, 768] Float
-- Error: matmul expects inner dims to match
-- weights has 512 cols, input has 1024 rows
result = matmul weights input
"#,
            ),
            correct_example: Some(
                r#"
weights :: Tensor '[768, 512] Float
input   :: Tensor '[768, 1024] Float
-- 768 == 768, so inner dimensions match
result = matmul (transpose weights) input  -- [512, 1024]

-- Or swap the order:
result = matmul input weights  -- Depends on what you want
"#,
            ),
        },
    );

    map.insert(
        "E0031",
        ErrorExplanation {
            code: "E0031",
            title: "Broadcast incompatible shapes",
            explanation: r#"
This error occurs when two tensors have shapes that cannot be broadcast
together according to NumPy-style broadcasting rules.

Broadcasting rules:
1. Shapes are compared element-wise from the trailing dimensions
2. Dimensions are compatible if they are equal or one of them is 1
3. Missing dimensions are treated as 1

For example:
  [3, 4] and [4] -> OK (becomes [3, 4])
  [3, 4] and [1, 4] -> OK (becomes [3, 4])
  [3, 4] and [2, 4] -> Error! (3 ≠ 2 and neither is 1)
"#,
            example: Some(
                r#"
a :: Tensor '[3, 4] Float
b :: Tensor '[2, 4] Float
result = a + b  -- Error: cannot broadcast [3,4] with [2,4]
"#,
            ),
            correct_example: Some(
                r#"
a :: Tensor '[3, 4] Float
b :: Tensor '[1, 4] Float
result = a + b  -- OK: b broadcasts to [3, 4]
"#,
            ),
        },
    );

    map.insert(
        "E0037",
        ErrorExplanation {
            code: "E0037",
            title: "Dynamic tensor conversion failed",
            explanation: r#"
This error occurs when attempting to convert a DynTensor to a statically-
shaped tensor with fromDynamic, but the runtime shape doesn't match.

DynTensor is an existentially-quantified wrapper that hides the shape:
  data DynTensor a where
    MkDynTensor :: Tensor shape a -> DynTensor a

When using fromDynamic, you must handle the case where the shapes don't
match:
  fromDynamic :: ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)

Always pattern match on the Maybe result to handle both cases.
"#,
            example: Some(
                r#"
processTensor :: DynTensor Float -> Tensor '[256, 256] Float
processTensor dyn = fromJust (fromDynamic witness dyn)
-- Error if runtime shape isn't [256, 256]!
"#,
            ),
            correct_example: Some(
                r#"
processTensor :: DynTensor Float -> Maybe (Tensor '[256, 256] Float)
processTensor dyn = case fromDynamic witness dyn of
    Just tensor -> Just (processStatic tensor)
    Nothing     -> Nothing  -- Handle shape mismatch
"#,
            ),
        },
    );

    // === Warnings (W0xxx) ===

    map.insert(
        "W0001",
        ErrorExplanation {
            code: "W0001",
            title: "Unused variable",
            explanation: r#"
This warning indicates that a variable is defined but never used.

While this doesn't prevent compilation, unused variables often indicate:
- Incomplete code (forgot to use the variable)
- Dead code that can be removed
- A typo in the variable name

To suppress this warning for intentionally unused variables, prefix the
name with an underscore: `_unused`.
"#,
            example: Some(
                r#"
foo x y = x + 1  -- Warning: y is unused
"#,
            ),
            correct_example: Some(
                r#"
foo x _y = x + 1  -- No warning: _y is intentionally unused
-- Or actually use y:
foo x y = x + y
"#,
            ),
        },
    );

    map
});

/// Look up an error explanation by code.
#[must_use]
pub fn get_explanation(code: &str) -> Option<&'static ErrorExplanation> {
    ERROR_REGISTRY.get(code)
}

/// Get all registered error codes.
#[must_use]
pub fn all_error_codes() -> Vec<&'static str> {
    let mut codes: Vec<_> = ERROR_REGISTRY.keys().copied().collect();
    codes.sort();
    codes
}

/// Format an error explanation for display.
#[must_use]
pub fn format_explanation(explanation: &ErrorExplanation) -> String {
    let mut output = String::new();

    output.push_str(&format!("# {} - {}\n\n", explanation.code, explanation.title));
    output.push_str(explanation.explanation.trim());
    output.push_str("\n\n");

    if let Some(example) = explanation.example {
        output.push_str("## Example of erroneous code:\n");
        output.push_str("```haskell");
        output.push_str(example);
        output.push_str("```\n\n");
    }

    if let Some(correct) = explanation.correct_example {
        output.push_str("## Corrected code:\n");
        output.push_str("```haskell");
        output.push_str(correct);
        output.push_str("```\n");
    }

    output
}

/// Print an error explanation to stdout.
pub fn print_explanation(code: &str) {
    match get_explanation(code) {
        Some(explanation) => {
            println!("{}", format_explanation(explanation));
        }
        None => {
            println!("Error code `{code}` not found.");
            println!("\nAvailable error codes:");
            for code in all_error_codes() {
                if let Some(exp) = get_explanation(code) {
                    println!("  {}: {}", code, exp.title);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_explanation() {
        let exp = get_explanation("E0001").unwrap();
        assert_eq!(exp.code, "E0001");
        assert_eq!(exp.title, "Type mismatch");
    }

    #[test]
    fn test_unknown_code() {
        assert!(get_explanation("E9999").is_none());
    }

    #[test]
    fn test_all_error_codes() {
        let codes = all_error_codes();
        assert!(!codes.is_empty());
        assert!(codes.contains(&"E0001"));
    }

    #[test]
    fn test_format_explanation() {
        let exp = get_explanation("E0001").unwrap();
        let formatted = format_explanation(exp);
        assert!(formatted.contains("E0001"));
        assert!(formatted.contains("Type mismatch"));
    }
}
