//! Pattern parsing.

use bhc_ast::{Expr, FieldPat, Lit, ModuleName, Pat, TyVar, Type};
use bhc_intern::{Ident, Symbol};
use bhc_lexer::TokenKind;
use bhc_span::Span;

use crate::{ParseError, ParseResult, Parser};

impl<'src> Parser<'src> {
    /// Check if the current token can start a pattern.
    pub fn is_pattern_start(&self) -> bool {
        match self.current_kind() {
            Some(kind) => matches!(
                kind,
                TokenKind::Ident(_)
                    | TokenKind::QualIdent(_, _)
                    | TokenKind::ConId(_)
                    | TokenKind::QualConId(_, _)
                    | TokenKind::IntLit(_)
                    | TokenKind::FloatLit(_)
                    | TokenKind::CharLit(_)
                    | TokenKind::StringLit(_)
                    | TokenKind::LParen
                    | TokenKind::LBracket
                    | TokenKind::Underscore
                    | TokenKind::Tilde
                    | TokenKind::Bang
                    | TokenKind::Minus
            ),
            None => false,
        }
    }

    /// Parse a pattern.
    pub fn parse_pattern(&mut self) -> ParseResult<Pat> {
        self.enter_recursion()?;
        let result = self.parse_infix_pattern();
        self.exit_recursion();
        result
    }

    /// Parse an infix pattern like `x : xs` or `x :| xs`.
    fn parse_infix_pattern(&mut self) -> ParseResult<Pat> {
        let mut pat = self.parse_app_pattern()?;

        while let Some(tok) = self.current() {
            match &tok.node.kind {
                // Constructor operators like `:`, `:|` are valid in infix patterns
                TokenKind::ConOperator(sym) => {
                    let op = Ident::new(*sym);
                    self.advance();
                    let rhs = self.parse_infix_pattern()?;
                    let span = pat.span().to(rhs.span());
                    pat = Pat::Infix(Box::new(pat), op, Box::new(rhs), span);
                }
                // Qualified constructor operators like `Seq.:>` in infix patterns
                TokenKind::QualConOperator(qual, sym) => {
                    let qualified_name =
                        Symbol::intern(&format!("{}.{}", qual.as_str(), sym.as_str()));
                    let op = Ident::new(qualified_name);
                    self.advance();
                    let rhs = self.parse_infix_pattern()?;
                    let span = pat.span().to(rhs.span());
                    pat = Pat::Infix(Box::new(pat), op, Box::new(rhs), span);
                }
                _ => break,
            }
        }

        Ok(pat)
    }

    /// Parse an application pattern like `Just x` or `W.Workspace i l ms`.
    fn parse_app_pattern(&mut self) -> ParseResult<Pat> {
        let first = self.parse_atom_pattern()?;

        // Check for constructor application
        match first {
            Pat::Con(con, args, span) if args.is_empty() => {
                let mut new_args = Vec::new();
                while self.is_apat_start() {
                    new_args.push(self.parse_atom_pattern()?);
                }
                if new_args.is_empty() {
                    return Ok(Pat::Con(con, args, span));
                }
                let new_span = span.to(new_args.last().unwrap().span());
                return Ok(Pat::Con(con, new_args, new_span));
            }
            Pat::Con(con, args, span) => {
                return Ok(Pat::Con(con, args, span));
            }
            Pat::QualCon(module_name, con, args, span) if args.is_empty() => {
                // Qualified constructor application like W.Workspace i l ms
                let mut new_args = Vec::new();
                while self.is_apat_start() {
                    new_args.push(self.parse_atom_pattern()?);
                }
                if new_args.is_empty() {
                    return Ok(Pat::QualCon(module_name, con, args, span));
                }
                let new_span = span.to(new_args.last().unwrap().span());
                return Ok(Pat::QualCon(module_name, con, new_args, new_span));
            }
            Pat::QualCon(module_name, con, args, span) => {
                return Ok(Pat::QualCon(module_name, con, args, span));
            }
            _ => {}
        }

        Ok(first)
    }

    /// Check if current token can start an atomic pattern.
    pub fn is_apat_start(&self) -> bool {
        match self.current_kind() {
            Some(kind) => matches!(
                kind,
                TokenKind::Ident(_)
                    | TokenKind::ConId(_)        // Constructors can be pattern arguments
                    | TokenKind::QualConId(_, _) // Qualified constructors too
                    | TokenKind::IntLit(_)
                    | TokenKind::FloatLit(_)
                    | TokenKind::CharLit(_)
                    | TokenKind::StringLit(_)
                    | TokenKind::LParen
                    | TokenKind::LBracket
                    | TokenKind::Underscore
                    | TokenKind::Tilde        // Lazy pattern ~x
                    | TokenKind::Bang // Strict pattern !x
            ),
            None => false,
        }
    }

    /// Parse an atomic pattern.
    /// This is used for function argument patterns in clause LHS.
    pub fn parse_atom_pattern(&mut self) -> ParseResult<Pat> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "pattern".to_string(),
        })?;

        match &tok.node.kind.clone() {
            TokenKind::Underscore => {
                let span = tok.span;
                self.advance();
                Ok(Pat::Wildcard(span))
            }

            TokenKind::Ident(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();

                // Check for as-pattern: x@pat
                if self.eat(&TokenKind::At) {
                    let pat = self.parse_atom_pattern()?;
                    let new_span = span.to(pat.span());
                    Ok(Pat::As(ident, Box::new(pat), new_span))
                } else {
                    Ok(Pat::Var(ident, span))
                }
            }

            TokenKind::QualIdent(qual, name) => {
                // Qualified identifier like M.x - treat as variable
                let full_name = format!("{}.{}", qual.as_str(), name.as_str());
                let ident = Ident::from_str(&full_name);
                let span = tok.span;
                self.advance();
                Ok(Pat::Var(ident, span))
            }

            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();

                // Check for record pattern: Con { field = pat, ... }
                if self.check(&TokenKind::LBrace) {
                    return self.parse_record_pattern(ident, span);
                }

                Ok(Pat::Con(ident, vec![], span))
            }

            TokenKind::QualConId(qual, name) => {
                // Qualified constructor like W.RationalRect
                let module_name = ModuleName {
                    parts: vec![*qual],
                    span: tok.span,
                };
                let ident = Ident::new(*name);
                let span = tok.span;
                self.advance();

                // Check for record pattern: Qual.Con { field = pat, ... }
                if self.check(&TokenKind::LBrace) {
                    return self.parse_qual_record_pattern(module_name, ident, span);
                }

                Ok(Pat::QualCon(module_name, ident, vec![], span))
            }

            TokenKind::Minus => {
                // Negative literal pattern: `-1`, `-2.5`. The lexer emits a
                // `Minus` token followed by the numeric literal, so consume
                // both here. Without this, a `-1 ->` case alternative (or a
                // negative-literal function-argument pattern) is a parse error,
                // whose recovery can silently drop the enclosing binding.
                let start = tok.span;
                self.advance(); // consume `-`
                let next = self
                    .current()
                    .map(|t| (t.node.kind.clone(), t.span))
                    .ok_or(ParseError::UnexpectedEof {
                        expected: "numeric literal after `-` in pattern".to_string(),
                    })?;
                match next {
                    (TokenKind::IntLit(lit), nspan) => {
                        let value = self.parse_int_literal(&lit.text, nspan)?;
                        self.advance();
                        Ok(Pat::Lit(Lit::Int(-value), start.to(nspan)))
                    }
                    (TokenKind::FloatLit(lit), nspan) => {
                        let value = self.parse_float_literal(&lit.text, nspan)?;
                        self.advance();
                        Ok(Pat::Lit(Lit::Float(-value), start.to(nspan)))
                    }
                    (other, nspan) => Err(ParseError::Unexpected {
                        found: other.description().to_string(),
                        expected: "numeric literal after `-` in pattern".to_string(),
                        span: nspan,
                    }),
                }
            }

            TokenKind::IntLit(ref lit) => {
                let span = tok.span;
                let value = self.parse_int_literal(&lit.text, span)?;
                self.advance();
                Ok(Pat::Lit(Lit::Int(value), span))
            }

            TokenKind::FloatLit(ref lit) => {
                let span = tok.span;
                let value = self.parse_float_literal(&lit.text, span)?;
                self.advance();
                Ok(Pat::Lit(Lit::Float(value), span))
            }

            TokenKind::CharLit(c) => {
                let span = tok.span;
                let c = *c;
                self.advance();
                Ok(Pat::Lit(Lit::Char(c), span))
            }

            TokenKind::StringLit(s) => {
                let span = tok.span;
                let s = s.clone();
                self.advance();
                Ok(Pat::Lit(Lit::String(s), span))
            }

            TokenKind::LParen => self.parse_paren_pattern(),

            TokenKind::LBracket => self.parse_list_pattern(),

            TokenKind::Tilde => {
                let start = tok.span;
                self.advance();
                let pat = self.parse_atom_pattern()?;
                let span = start.to(pat.span());
                Ok(Pat::Lazy(Box::new(pat), span))
            }

            TokenKind::Bang => {
                let start = tok.span;
                self.advance();
                let pat = self.parse_atom_pattern()?;
                let span = start.to(pat.span());
                Ok(Pat::Bang(Box::new(pat), span))
            }

            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "pattern".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse a parenthesized pattern, tuple pattern, or view pattern.
    fn parse_paren_pattern(&mut self) -> ParseResult<Pat> {
        let start = self.current_span();
        self.expect(&TokenKind::LParen)?;

        if self.eat(&TokenKind::RParen) {
            // Unit pattern: ()
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Pat::Con(Ident::from_str("()"), vec![], span));
        }

        // Parse the first element, allowing a view pattern (`expr -> pat`).
        // Inside parens the surrounding delimiter disambiguates the `->`, so
        // view patterns are legal both standalone `(e -> p)` and as tuple
        // elements `(l, e -> p)`.
        let first = self.parse_pattern_or_view()?;

        // Check for pattern type signature: (pat :: Type)
        if self.eat(&TokenKind::DoubleColon) {
            let ty = self.parse_type()?;
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            return Ok(Pat::Ann(Box::new(first), ty, span));
        }

        if self.eat(&TokenKind::Comma) {
            // Tuple pattern; each element may itself be a view pattern.
            let mut pats = vec![first];
            loop {
                pats.push(self.parse_pattern_or_view()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Pat::Tuple(pats, span))
        } else {
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            match first {
                // A standalone view pattern is returned as-is.
                Pat::View(..) => Ok(first),
                // Otherwise it's an ordinary parenthesized pattern.
                _ => Ok(Pat::Paren(Box::new(first), span)),
            }
        }
    }

    /// Parse a pattern that may be a view pattern (`expr -> pat` or
    /// `f x -> pat`). Only used in delimited contexts (parenthesized /
    /// tuple / list elements) where a surrounding delimiter disambiguates
    /// the `->` from, e.g., a case-alternative arrow. In an undelimited
    /// context (`parse_pattern` used by `parse_alt`) the `->` must remain the
    /// alternative separator, which is why this handling is not folded into
    /// `parse_pattern` itself.
    fn parse_pattern_or_view(&mut self) -> ParseResult<Pat> {
        let start = self.current_span();
        let first = self.parse_pattern()?;

        // Simple view pattern: `expr -> pat`
        if self.check(&TokenKind::Arrow) {
            self.advance();
            let view_expr = self.pat_to_expr(&first)?;
            let result_pat = self.parse_pattern()?;
            let span = start.to(result_pat.span());
            return Ok(Pat::View(Box::new(view_expr), Box::new(result_pat), span));
        }

        // Type-applied view expression: `TR.decimal @Integer -> pat`. After a
        // QUALIFIED var the pattern parser leaves the `@` pending (only the
        // unqualified Ident arm folds it into an as-pattern), so rebuild the
        // type application(s) here and require the view arrow.
        if self.check(&TokenKind::At) && matches!(&first, Pat::Var(..)) {
            let save_pos = self.pos;
            let mut view_expr = self.pat_to_expr(&first)?;
            let mut applied = true;
            while self.eat(&TokenKind::At) {
                match self.parse_atype() {
                    Ok(ty) => {
                        let sp = view_expr.span().to(ty.span());
                        view_expr = Expr::TypeApp(Box::new(view_expr), ty, sp);
                    }
                    Err(_) => {
                        applied = false;
                        break;
                    }
                }
            }
            if applied && self.eat(&TokenKind::Arrow) {
                let result_pat = self.parse_pattern()?;
                let span = start.to(result_pat.span());
                return Ok(Pat::View(Box::new(view_expr), Box::new(result_pat), span));
            }
            self.pos = save_pos;
        }

        // Operator-continued view expression: `f . g -> pat` — e.g.
        // `(normalise . unEscapeString -> path)` in Readers.EPUB. In a
        // delimited pattern context no non-constructor operator can continue a
        // pattern, so the operator must belong to the view expression: convert
        // the prefix and let the expression parser take over up to the arrow.
        if matches!(
            self.current_kind(),
            Some(TokenKind::Operator(_) | TokenKind::Dot)
        ) {
            let save_pos = self.pos;
            if let Ok(prefix) = self.pat_to_expr(&first) {
                if let Ok(view_expr) = self.continue_infix_expr(prefix, 0) {
                    if self.eat(&TokenKind::Arrow) {
                        let result_pat = self.parse_pattern()?;
                        let span = start.to(result_pat.span());
                        return Ok(Pat::View(Box::new(view_expr), Box::new(result_pat), span));
                    }
                }
            }
            self.pos = save_pos;
        }

        // Applied view pattern: `f x y -> pat`
        if matches!(&first, Pat::Var(..)) && self.is_apat_start() {
            let save_pos = self.pos;
            let mut args: Vec<Pat> = Vec::new();
            while self.is_apat_start() && !self.check(&TokenKind::Arrow) {
                args.push(self.parse_atom_pattern()?);
            }
            if self.eat(&TokenKind::Arrow) {
                let mut view_expr = self.pat_to_expr(&first)?;
                for arg in &args {
                    let arg_expr = self.pat_to_expr(arg)?;
                    let new_span = view_expr.span().to(arg_expr.span());
                    view_expr = Expr::App(Box::new(view_expr), Box::new(arg_expr), new_span);
                }
                let result_pat = self.parse_pattern()?;
                let span = start.to(result_pat.span());
                return Ok(Pat::View(Box::new(view_expr), Box::new(result_pat), span));
            }
            // Not a view pattern — backtrack.
            self.pos = save_pos;
        }

        // Pattern type annotation in a delimited context: `(n :: Int, fp)`
        // (Writers.EPUB's comprehension binder). Without this the `::` hits
        // the closing-delimiter check and the whole binding is dropped.
        if self.check(&TokenKind::DoubleColon) {
            self.advance();
            let ty = self.parse_type()?;
            let span = first.span().to(ty.span());
            return Ok(Pat::Ann(Box::new(first), ty, span));
        }

        Ok(first)
    }

    /// Convert a pattern to an expression (for view patterns).
    /// This handles the common case where the "pattern" before -> is actually a function.
    fn pat_to_expr(&self, pat: &Pat) -> ParseResult<Expr> {
        use bhc_ast::Expr;
        match pat {
            Pat::Var(name, span) => {
                // Check if this is a qualified variable (e.g. "L.reverse" from QualIdent token)
                let name_str = name.name.as_str();
                if let Some(dot_pos) = name_str.rfind('.') {
                    let qualifier = &name_str[..dot_pos];
                    let local = &name_str[dot_pos + 1..];
                    if !qualifier.is_empty() && !local.is_empty() {
                        let module_name = ModuleName {
                            parts: vec![Symbol::intern(qualifier)],
                            span: *span,
                        };
                        let local_ident = Ident::from_str(local);
                        return Ok(Expr::QualVar(module_name, local_ident, *span));
                    }
                }
                Ok(Expr::Var(*name, *span))
            }
            Pat::Con(name, args, span) => {
                if args.is_empty() {
                    Ok(Expr::Con(*name, *span))
                } else {
                    // Constructor application: Con a b -> App (App Con a) b
                    let mut result = Expr::Con(*name, *span);
                    for arg in args {
                        let arg_expr = self.pat_to_expr(arg)?;
                        let new_span = result.span().to(arg_expr.span());
                        result = Expr::App(Box::new(result), Box::new(arg_expr), new_span);
                    }
                    Ok(result)
                }
            }
            Pat::QualCon(module_name, name, args, span) => {
                if args.is_empty() {
                    Ok(Expr::QualCon(module_name.clone(), *name, *span))
                } else {
                    // Constructor application: Mod.Con a b -> App (App Mod.Con a) b
                    let mut result = Expr::QualCon(module_name.clone(), *name, *span);
                    for arg in args {
                        let arg_expr = self.pat_to_expr(arg)?;
                        let new_span = result.span().to(arg_expr.span());
                        result = Expr::App(Box::new(result), Box::new(arg_expr), new_span);
                    }
                    Ok(result)
                }
            }
            Pat::Lit(lit, span) => Ok(Expr::Lit(lit.clone(), *span)),
            Pat::Paren(inner, span) => {
                let inner_expr = self.pat_to_expr(inner)?;
                Ok(Expr::Paren(Box::new(inner_expr), *span))
            }
            Pat::Tuple(elems, span) => {
                let mut exprs = Vec::new();
                for elem in elems {
                    exprs.push(self.pat_to_expr(elem)?);
                }
                Ok(Expr::Tuple(exprs, *span))
            }
            Pat::List(elems, span) => {
                let mut exprs = Vec::new();
                for elem in elems {
                    exprs.push(self.pat_to_expr(elem)?);
                }
                Ok(Expr::List(exprs, *span))
            }
            Pat::Wildcard(span) => {
                // Wildcard in view expression context — shouldn't happen, but handle gracefully
                Ok(Expr::Var(Ident::from_str("_"), *span))
            }
            Pat::As(name, sub, span) => {
                // Not a real as-pattern: in a view pattern's EXPRESSION position
                // an `@` is a visible type application — `(TR.decimal @Integer
                // -> Right (x, "")) = ...` (Readers.Pod's `entity`). The pattern
                // parser reads `f @T` as `f @ T` before the `->` is seen, so
                // rebuild the type application here when the sub-pattern is
                // shaped like a type. (A genuine as-pattern before `->` is not
                // an expression, so nothing valid is lost.)
                let base = self.pat_to_expr(&Pat::Var(*name, *span))?;
                let ty = Self::pat_to_type(sub).ok_or_else(|| ParseError::Unexpected {
                    found: "as-pattern".to_string(),
                    expected: "simple expression for view pattern".to_string(),
                    span: pat.span(),
                })?;
                Ok(Expr::TypeApp(Box::new(base), ty, *span))
            }
            _ => Err(ParseError::Unexpected {
                found: "complex pattern".to_string(),
                expected: "simple expression for view pattern".to_string(),
                span: pat.span(),
            }),
        }
    }

    /// Convert a pattern that syntactically denotes a TYPE back into one.
    ///
    /// Used by `pat_to_expr` for view-pattern expressions where `f @T` was
    /// mis-read as an as-pattern: the "sub-pattern" after the `@` is really a
    /// type argument. Returns `None` for shapes that can't be a type (literals,
    /// wildcards, records, ...), which surfaces the original parse error.
    fn pat_to_type(pat: &Pat) -> Option<Type> {
        match pat {
            // Uppercase names parse as (possibly applied) constructor patterns.
            Pat::Con(name, args, span) => {
                let mut ty = Type::Con(*name, *span);
                for arg in args {
                    let arg_ty = Self::pat_to_type(arg)?;
                    let sp = ty.span().to(arg_ty.span());
                    ty = Type::App(Box::new(ty), Box::new(arg_ty), sp);
                }
                Some(ty)
            }
            Pat::QualCon(module, name, args, span) => {
                let mut ty = Type::QualCon(module.clone(), *name, *span);
                for arg in args {
                    let arg_ty = Self::pat_to_type(arg)?;
                    let sp = ty.span().to(arg_ty.span());
                    ty = Type::App(Box::new(ty), Box::new(arg_ty), sp);
                }
                Some(ty)
            }
            // Lowercase names are type variables (`f @a`).
            Pat::Var(name, span) => Some(Type::Var(
                TyVar {
                    name: *name,
                    span: *span,
                },
                *span,
            )),
            Pat::Paren(inner, span) => {
                Some(Type::Paren(Box::new(Self::pat_to_type(inner)?), *span))
            }
            Pat::Tuple(elems, span) => {
                let tys = elems
                    .iter()
                    .map(Self::pat_to_type)
                    .collect::<Option<Vec<_>>>()?;
                Some(Type::Tuple(tys, *span))
            }
            Pat::List(elems, span) if elems.len() == 1 => {
                Some(Type::List(Box::new(Self::pat_to_type(&elems[0])?), *span))
            }
            _ => None,
        }
    }

    /// Parse a list pattern.
    fn parse_list_pattern(&mut self) -> ParseResult<Pat> {
        let start = self.current_span();
        self.expect(&TokenKind::LBracket)?;

        if self.eat(&TokenKind::RBracket) {
            // Empty list: []
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Pat::List(vec![], span));
        }

        // Each element may be a view pattern (`[l, e -> p]`); the brackets
        // disambiguate the `->` just as parens do for tuples.
        let mut pats = vec![self.parse_pattern_or_view()?];
        while self.eat(&TokenKind::Comma) {
            if self.check(&TokenKind::RBracket) {
                break;
            }
            pats.push(self.parse_pattern_or_view()?);
        }

        let end = self.expect(&TokenKind::RBracket)?;
        let span = start.to(end.span);
        Ok(Pat::List(pats, span))
    }

    /// Parse a record pattern: `Con { field = pat, ... }` or `Con { field = pat, .. }`
    fn parse_record_pattern(&mut self, con: Ident, start: Span) -> ParseResult<Pat> {
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        let mut has_wildcard = false;
        if !self.check(&TokenKind::RBrace) {
            if self.eat(&TokenKind::DotDot) {
                has_wildcard = true;
            } else {
                fields.push(self.parse_field_pat()?);
                while self.eat(&TokenKind::Comma) {
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                    if self.eat(&TokenKind::DotDot) {
                        has_wildcard = true;
                        break;
                    }
                    fields.push(self.parse_field_pat()?);
                }
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;
        let span = start.to(end.span);
        Ok(Pat::Record(con, fields, has_wildcard, span))
    }

    /// Parse a qualified record pattern: `Qual.Con { field = pat, ... }` or `Qual.Con { .. }`
    fn parse_qual_record_pattern(
        &mut self,
        module_name: ModuleName,
        con: Ident,
        start: Span,
    ) -> ParseResult<Pat> {
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        let mut has_wildcard = false;
        if !self.check(&TokenKind::RBrace) {
            if self.eat(&TokenKind::DotDot) {
                has_wildcard = true;
            } else {
                fields.push(self.parse_field_pat()?);
                while self.eat(&TokenKind::Comma) {
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                    if self.eat(&TokenKind::DotDot) {
                        has_wildcard = true;
                        break;
                    }
                    fields.push(self.parse_field_pat()?);
                }
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;
        let span = start.to(end.span);
        Ok(Pat::QualRecord(
            module_name,
            con,
            fields,
            has_wildcard,
            span,
        ))
    }

    /// Parse a field pattern: `field = pat`, `Mod.field = pat`, or `field` (punning)
    fn parse_field_pat(&mut self) -> ParseResult<FieldPat> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "field name".to_string(),
        })?;

        let (qualifier, name, span) = match &tok.node.kind {
            TokenKind::Ident(sym) => (None, Ident::new(*sym), tok.span),
            TokenKind::QualIdent(qual, sym) => {
                let module_name = ModuleName {
                    parts: vec![*qual],
                    span: tok.span,
                };
                (Some(module_name), Ident::new(*sym), tok.span)
            }
            _ => {
                return Err(ParseError::Unexpected {
                    found: tok.node.kind.description().to_string(),
                    expected: "field name".to_string(),
                    span: tok.span,
                });
            }
        };
        self.advance();

        let pat = if self.eat(&TokenKind::Eq) {
            Some(self.parse_pattern()?)
        } else {
            None // Punning: `Foo { bar }` means `Foo { bar = bar }`
        };

        let end_span = pat.as_ref().map(|p| p.span()).unwrap_or(span);
        let full_span = span.to(end_span);
        Ok(FieldPat {
            qualifier,
            name,
            pat,
            span: full_span,
        })
    }
}
