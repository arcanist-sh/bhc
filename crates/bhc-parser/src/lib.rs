//! Parser for Haskell 2026 source code.
//!
//! This crate provides a recursive descent parser that produces an AST
//! from a token stream.

#![warn(missing_docs)]

use bhc_ast::{Decl, Expr, Module, Pat, Type};
use bhc_diagnostics::{Diagnostic, DiagnosticHandler, FullSpan};
use bhc_lexer::{Lexer, Token, TokenKind};
use bhc_span::{FileId, Span, Spanned};
use thiserror::Error;

mod expr;
mod decl;
mod pattern;
mod types;

/// Parser error type.
#[derive(Debug, Error)]
pub enum ParseError {
    /// Unexpected token.
    #[error("unexpected {found}, expected {expected}")]
    Unexpected {
        /// What was found.
        found: String,
        /// What was expected.
        expected: String,
        /// Location.
        span: Span,
    },

    /// Unexpected end of file.
    #[error("unexpected end of file")]
    UnexpectedEof {
        /// What was expected.
        expected: String,
    },

    /// Invalid literal.
    #[error("invalid literal: {message}")]
    InvalidLiteral {
        /// Error message.
        message: String,
        /// Location.
        span: Span,
    },
}

impl ParseError {
    /// Convert to a diagnostic.
    #[must_use]
    pub fn to_diagnostic(&self, file: FileId) -> Diagnostic {
        match self {
            Self::Unexpected {
                found,
                expected,
                span,
            } => Diagnostic::error(format!("unexpected {found}, expected {expected}"))
                .with_label(FullSpan::new(file, *span), "unexpected token here"),
            Self::UnexpectedEof { expected } => {
                Diagnostic::error(format!("unexpected end of file, expected {expected}"))
            }
            Self::InvalidLiteral { message, span } => {
                Diagnostic::error(format!("invalid literal: {message}"))
                    .with_label(FullSpan::new(file, *span), "invalid literal")
            }
        }
    }
}

/// The result of parsing.
pub type ParseResult<T> = Result<T, ParseError>;

/// A parser for Haskell 2026 source code.
pub struct Parser<'src> {
    /// The token stream.
    tokens: Vec<Spanned<Token>>,
    /// Current position in the token stream.
    pos: usize,
    /// Diagnostic handler.
    diagnostics: DiagnosticHandler,
    /// Source file ID.
    file_id: FileId,
    /// The source code (for error messages).
    #[allow(dead_code)]
    src: &'src str,
}

impl<'src> Parser<'src> {
    /// Create a new parser for the given source code.
    #[must_use]
    pub fn new(src: &'src str, file_id: FileId) -> Self {
        let tokens: Vec<_> = Lexer::new(src).collect();
        Self {
            tokens,
            pos: 0,
            diagnostics: DiagnosticHandler::new(),
            file_id,
            src,
        }
    }

    /// Get the current token.
    fn current(&self) -> Option<&Spanned<Token>> {
        self.tokens.get(self.pos)
    }

    /// Get the current token kind.
    fn current_kind(&self) -> Option<&TokenKind> {
        self.current().map(|t| &t.node.kind)
    }

    /// Get the current span.
    fn current_span(&self) -> Span {
        self.current()
            .map(|t| t.span)
            .unwrap_or(Span::DUMMY)
    }

    /// Check if we're at the end of input.
    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Advance to the next token.
    fn advance(&mut self) -> Option<Spanned<Token>> {
        if self.at_eof() {
            None
        } else {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(tok)
        }
    }

    /// Check if the current token matches the given kind.
    fn check(&self, kind: &TokenKind) -> bool {
        self.current_kind() == Some(kind)
    }

    /// Check if the current token is a constructor identifier.
    fn check_con_id(&self) -> bool {
        matches!(self.current_kind(), Some(TokenKind::ConId(_)))
    }

    /// Check if the current token is an identifier.
    fn check_ident(&self) -> bool {
        matches!(self.current_kind(), Some(TokenKind::Ident(_)))
    }

    /// Consume a token if it matches the given kind.
    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Expect a token of the given kind.
    fn expect(&mut self, kind: &TokenKind) -> ParseResult<Spanned<Token>> {
        if self.check(kind) {
            Ok(self.advance().unwrap())
        } else if self.at_eof() {
            Err(ParseError::UnexpectedEof {
                expected: kind.description().to_string(),
            })
        } else {
            let current = self.current().unwrap();
            Err(ParseError::Unexpected {
                found: current.node.kind.description().to_string(),
                expected: kind.description().to_string(),
                span: current.span,
            })
        }
    }

    /// Emit a diagnostic.
    fn emit(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.emit(diagnostic);
    }

    /// Check if there are errors.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }

    /// Take the diagnostics.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.diagnostics.take_diagnostics()
    }
}

/// Parse a module from source code.
pub fn parse_module(src: &str, file_id: FileId) -> (Option<Module>, Vec<Diagnostic>) {
    let mut parser = Parser::new(src, file_id);
    let module = parser.parse_module();
    let diagnostics = parser.take_diagnostics();

    match module {
        Ok(m) => (Some(m), diagnostics),
        Err(e) => {
            let mut diags = diagnostics;
            diags.push(e.to_diagnostic(file_id));
            (None, diags)
        }
    }
}

/// Parse an expression from source code.
pub fn parse_expr(src: &str, file_id: FileId) -> (Option<Expr>, Vec<Diagnostic>) {
    let mut parser = Parser::new(src, file_id);
    let expr = parser.parse_expr();
    let diagnostics = parser.take_diagnostics();

    match expr {
        Ok(e) => (Some(e), diagnostics),
        Err(e) => {
            let mut diags = diagnostics;
            diags.push(e.to_diagnostic(file_id));
            (None, diags)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = Parser::new("let x = 1 in x", FileId::new(0));
        assert!(!parser.at_eof());
    }
}
