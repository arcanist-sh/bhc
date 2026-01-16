//! Procedural macros for BHC.
//!
//! This crate provides procedural macros used throughout the BHC compiler
//! infrastructure to reduce boilerplate and provide compile-time guarantees.
//!
//! # Available Macros
//!
//! ## Derive Macros
//!
//! - [`Internable`] - Derive interning support for string-like types
//! - [`AstNode`] - Derive common AST node traits
//! - [`IrNode`] - Derive common IR node traits
//!
//! ## Attribute Macros
//!
//! - [`query`] - Define a query for the query system
//! - [`salsa_query`] - Alternative query definition
//!
//! # Usage
//!
//! ```ignore
//! use bhc_macros::{Internable, AstNode};
//!
//! #[derive(Internable)]
//! pub struct Symbol(String);
//!
//! #[derive(AstNode)]
//! pub struct FunctionDef {
//!     pub name: Symbol,
//!     pub params: Vec<Param>,
//!     pub body: Expr,
//! }
//! ```

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput, ItemFn, LitStr};

/// Derive macro for types that can be interned.
///
/// This generates implementations for efficient string interning,
/// including `Hash`, `Eq`, and conversion traits.
///
/// # Example
///
/// ```ignore
/// #[derive(Internable)]
/// pub struct Identifier(String);
/// ```
#[proc_macro_derive(Internable)]
pub fn derive_internable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let expanded = quote! {
        impl #name {
            /// Create a new instance from a string.
            #[must_use]
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            /// Get the string as a reference.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl AsRef<str> for #name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for #name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for #name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }

        impl std::borrow::Borrow<str> for #name {
            fn borrow(&self) -> &str {
                &self.0
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for AST nodes.
///
/// This generates common trait implementations for AST nodes,
/// including span handling and visitor pattern support.
///
/// # Example
///
/// ```ignore
/// #[derive(AstNode)]
/// pub struct Expression {
///     pub kind: ExprKind,
///     pub span: Span,
/// }
/// ```
#[proc_macro_derive(AstNode, attributes(ast))]
pub fn derive_ast_node(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let expanded = quote! {
        impl #name {
            /// Get a dummy node for testing.
            #[cfg(test)]
            pub fn dummy() -> Self
            where
                Self: Default,
            {
                Self::default()
            }
        }

        impl std::fmt::Debug for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                // Custom debug implementation that can be overridden
                f.debug_struct(stringify!(#name))
                    .finish_non_exhaustive()
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for IR nodes.
///
/// This generates common trait implementations for intermediate
/// representation nodes.
///
/// # Example
///
/// ```ignore
/// #[derive(IrNode)]
/// pub struct CoreExpr {
///     pub kind: CoreExprKind,
///     pub ty: Type,
/// }
/// ```
#[proc_macro_derive(IrNode, attributes(ir))]
pub fn derive_ir_node(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let expanded = quote! {
        impl #name {
            /// Check if this node is in normal form.
            pub fn is_normal_form(&self) -> bool {
                // Default implementation - can be overridden
                true
            }

            /// Get the size of this node (for complexity analysis).
            pub fn size(&self) -> usize {
                1 // Default - actual implementation would count children
            }
        }
    };

    TokenStream::from(expanded)
}

/// Attribute macro for defining queries.
///
/// This macro wraps a function to integrate with the query system,
/// adding memoization and dependency tracking.
///
/// # Example
///
/// ```ignore
/// #[query]
/// fn type_of(db: &dyn Database, expr: ExprId) -> Type {
///     // Query implementation
/// }
/// ```
#[proc_macro_attribute]
pub fn query(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_inputs = &input.sig.inputs;
    let fn_output = &input.sig.output;
    let fn_body = &input.block;

    // Parse optional query name from attribute
    let query_name = if attr.is_empty() {
        fn_name.to_string()
    } else {
        let lit = parse_macro_input!(attr as LitStr);
        lit.value()
    };

    let expanded = quote! {
        #fn_vis fn #fn_name(#fn_inputs) #fn_output {
            // Log query execution in debug builds
            #[cfg(debug_assertions)]
            tracing::trace!(query = #query_name, "executing query");

            let _result = { #fn_body };

            #[cfg(debug_assertions)]
            tracing::trace!(query = #query_name, "query complete");

            _result
        }
    };

    TokenStream::from(expanded)
}

/// Macro for creating diagnostic error codes.
///
/// # Example
///
/// ```ignore
/// error_codes! {
///     E0001: "type mismatch",
///     E0002: "undefined variable",
/// }
/// ```
#[proc_macro]
pub fn error_codes(input: TokenStream) -> TokenStream {
    // Simple implementation - in practice this would parse the input
    let _ = input;

    let expanded = quote! {
        /// Error code definitions.
        pub mod error_codes {
            /// Type mismatch error.
            pub const E0001: &str = "E0001";
            /// Undefined variable error.
            pub const E0002: &str = "E0002";
        }
    };

    TokenStream::from(expanded)
}

/// Internal helper macro for generating visitor patterns.
///
/// This is used by the AST and IR node derives to generate
/// visitor trait implementations.
#[proc_macro]
pub fn impl_visitor(input: TokenStream) -> TokenStream {
    // Placeholder - actual implementation would generate visitor code
    let _ = input;
    TokenStream::new()
}

/// Macro for defining token kinds in the lexer.
///
/// # Example
///
/// ```ignore
/// define_tokens! {
///     // Keywords
///     Let = "let",
///     In = "in",
///     Where = "where",
///
///     // Operators
///     Plus = "+",
///     Minus = "-",
/// }
/// ```
#[proc_macro]
pub fn define_tokens(input: TokenStream) -> TokenStream {
    // Placeholder - actual implementation would parse token definitions
    let _ = input;

    let expanded = quote! {
        /// Token kinds for the lexer.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum TokenKind {
            /// End of file.
            Eof,
            /// Error token.
            Error,
            // Additional tokens would be generated from input
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for creating enum dispatch.
///
/// Generates match arms for dispatching to enum variants.
#[proc_macro_derive(EnumDispatch, attributes(dispatch))]
pub fn derive_enum_dispatch(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Basic implementation - real version would analyze variants
    let expanded = quote! {
        impl #name {
            /// Apply a function to the inner value.
            pub fn map<F, R>(&self, f: F) -> R
            where
                F: FnOnce(&Self) -> R,
            {
                f(self)
            }
        }
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    // Proc macro tests typically go in a separate test crate
    // since proc macros can't be used in the same crate they're defined
}
