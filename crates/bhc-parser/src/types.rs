//! Type parsing.

use bhc_ast::*;
use bhc_intern::Ident;
use bhc_lexer::TokenKind;

use crate::{ParseResult, Parser, ParseError};

impl<'src> Parser<'src> {
    /// Parse a type.
    pub fn parse_type(&mut self) -> ParseResult<Type> {
        self.parse_fun_type()
    }

    /// Parse a function type: `a -> b`.
    fn parse_fun_type(&mut self) -> ParseResult<Type> {
        let lhs = self.parse_app_type()?;

        if self.eat(&TokenKind::Arrow) {
            let rhs = self.parse_fun_type()?;
            let span = lhs.span().to(rhs.span());
            Ok(Type::Fun(Box::new(lhs), Box::new(rhs), span))
        } else {
            Ok(lhs)
        }
    }

    /// Parse a type application: `Maybe Int`.
    fn parse_app_type(&mut self) -> ParseResult<Type> {
        let mut ty = self.parse_atype()?;

        while self.is_atype_start() {
            let arg = self.parse_atype()?;
            let span = ty.span().to(arg.span());
            ty = Type::App(Box::new(ty), Box::new(arg), span);
        }

        Ok(ty)
    }

    /// Check if current token can start an atomic type.
    pub fn is_atype_start(&self) -> bool {
        match self.current_kind() {
            Some(kind) => matches!(
                kind,
                TokenKind::Ident(_)
                    | TokenKind::ConId(_)
                    | TokenKind::LParen
                    | TokenKind::LBracket
            ),
            None => false,
        }
    }

    /// Parse an atomic type.
    pub fn parse_atype(&mut self) -> ParseResult<Type> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "type".to_string(),
        })?;

        match &tok.node.kind.clone() {
            TokenKind::Ident(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                Ok(Type::Var(TyVar { name: ident, span }, span))
            }

            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                Ok(Type::Con(ident, span))
            }

            TokenKind::LParen => self.parse_paren_type(),

            TokenKind::LBracket => self.parse_list_type(),

            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "type".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse a parenthesized type or tuple type.
    fn parse_paren_type(&mut self) -> ParseResult<Type> {
        let start = self.current_span();
        self.expect(&TokenKind::LParen)?;

        if self.eat(&TokenKind::RParen) {
            // Unit type: ()
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Type::Tuple(vec![], span));
        }

        // Check for function type in parens: (->)
        if self.eat(&TokenKind::Arrow) {
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            return Ok(Type::Con(Ident::from_str("->"), span));
        }

        let first = self.parse_type()?;

        if self.eat(&TokenKind::Comma) {
            // Tuple type
            let mut types = vec![first];
            loop {
                types.push(self.parse_type()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Type::Tuple(types, span))
        } else {
            // Parenthesized type
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Type::Paren(Box::new(first), span))
        }
    }

    /// Parse a list type: `[a]`.
    fn parse_list_type(&mut self) -> ParseResult<Type> {
        let start = self.current_span();
        self.expect(&TokenKind::LBracket)?;

        if self.eat(&TokenKind::RBracket) {
            // List type constructor: []
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Type::Con(Ident::from_str("[]"), span));
        }

        let elem = self.parse_type()?;
        let end = self.expect(&TokenKind::RBracket)?;
        let span = start.to(end.span);

        Ok(Type::List(Box::new(elem), span))
    }

    /// Parse a forall type.
    #[allow(dead_code)]
    fn parse_forall_type(&mut self) -> ParseResult<Type> {
        let start = self.current_span();
        self.expect(&TokenKind::Forall)?;

        let mut vars = Vec::new();
        while let Some(tok) = self.current() {
            match &tok.node.kind {
                TokenKind::Ident(sym) => {
                    let name = Ident::new(*sym);
                    let span = tok.span;
                    self.advance();
                    vars.push(TyVar { name, span });
                }
                TokenKind::Operator(s) if s.as_str() == "." => {
                    self.advance();
                    break;
                }
                _ => break,
            }
        }

        let ty = self.parse_type()?;
        let span = start.to(ty.span());

        Ok(Type::Forall(vars, Box::new(ty), span))
    }
}
