//! Lexical analysis for BHC.
//!
//! This crate provides a fast lexer for Haskell 2026 source code,
//! producing a stream of tokens with source locations.

#![warn(missing_docs)]

use bhc_intern::Symbol;
use bhc_span::{BytePos, Span, Spanned};

mod token;

pub use token::{Token, TokenKind};

/// A lexer for Haskell 2026 source code.
pub struct Lexer<'src> {
    src: &'src str,
    pos: usize,
    /// Current indentation levels for layout rule
    indent_stack: Vec<u32>,
    /// Pending tokens from layout rule
    pending: Vec<Spanned<Token>>,
    /// Whether we're at the start of a line
    at_line_start: bool,
    /// Current line's indentation
    current_indent: u32,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given source code.
    #[must_use]
    pub fn new(src: &'src str) -> Self {
        Self {
            src,
            pos: 0,
            indent_stack: vec![0],
            pending: Vec::new(),
            at_line_start: true,
            current_indent: 0,
        }
    }

    /// Get the remaining source code.
    fn remaining(&self) -> &'src str {
        &self.src[self.pos..]
    }

    /// Peek at the next character without consuming it.
    fn peek(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    /// Peek at the character after next.
    fn peek2(&self) -> Option<char> {
        let mut chars = self.remaining().chars();
        chars.next();
        chars.next()
    }

    /// Advance by one character.
    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Advance while a predicate is true.
    fn advance_while(&mut self, pred: impl Fn(char) -> bool) {
        while let Some(c) = self.peek() {
            if pred(c) {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip whitespace and comments, returning whether a newline was encountered.
    fn skip_whitespace(&mut self) -> bool {
        let mut saw_newline = false;
        loop {
            match self.peek() {
                Some(' ' | '\t') => {
                    self.advance();
                }
                Some('\n') => {
                    self.advance();
                    saw_newline = true;
                    self.at_line_start = true;
                    self.current_indent = 0;
                    // Count indentation
                    while let Some(' ') = self.peek() {
                        self.advance();
                        self.current_indent += 1;
                    }
                    while let Some('\t') = self.peek() {
                        self.advance();
                        self.current_indent += 8; // Tab = 8 spaces
                    }
                }
                Some('\r') => {
                    self.advance();
                }
                Some('-') if self.peek2() == Some('-') => {
                    // Line comment
                    self.advance();
                    self.advance();
                    self.advance_while(|c| c != '\n');
                }
                Some('{') if self.peek2() == Some('-') => {
                    // Block comment
                    self.skip_block_comment();
                }
                _ => break,
            }
        }
        saw_newline
    }

    /// Skip a block comment, handling nesting.
    fn skip_block_comment(&mut self) {
        self.advance(); // {
        self.advance(); // -

        let mut depth = 1;
        while depth > 0 {
            match self.peek() {
                Some('{') if self.peek2() == Some('-') => {
                    self.advance();
                    self.advance();
                    depth += 1;
                }
                Some('-') if self.peek2() == Some('}') => {
                    self.advance();
                    self.advance();
                    depth -= 1;
                }
                Some(_) => {
                    self.advance();
                }
                None => break, // Unterminated comment
            }
        }
    }

    /// Lex an identifier or keyword.
    fn lex_ident(&mut self, start: usize) -> Token {
        self.advance_while(|c| c.is_alphanumeric() || c == '_' || c == '\'');
        let text = &self.src[start..self.pos];
        let sym = Symbol::intern(text);

        // Check for keywords
        let kind = match text {
            "case" => TokenKind::Case,
            "class" => TokenKind::Class,
            "data" => TokenKind::Data,
            "default" => TokenKind::Default,
            "deriving" => TokenKind::Deriving,
            "do" => TokenKind::Do,
            "else" => TokenKind::Else,
            "forall" => TokenKind::Forall,
            "foreign" => TokenKind::Foreign,
            "if" => TokenKind::If,
            "import" => TokenKind::Import,
            "in" => TokenKind::In,
            "infix" => TokenKind::Infix,
            "infixl" => TokenKind::Infixl,
            "infixr" => TokenKind::Infixr,
            "instance" => TokenKind::Instance,
            "let" => TokenKind::Let,
            "module" => TokenKind::Module,
            "newtype" => TokenKind::Newtype,
            "of" => TokenKind::Of,
            "qualified" => TokenKind::Qualified,
            "then" => TokenKind::Then,
            "type" => TokenKind::Type,
            "where" => TokenKind::Where,
            "lazy" => TokenKind::Lazy,
            "strict" => TokenKind::Strict,
            "_" => TokenKind::Underscore,
            _ => TokenKind::Ident(sym),
        };

        Token { kind }
    }

    /// Lex a constructor (uppercase identifier).
    fn lex_conid(&mut self, start: usize) -> Token {
        self.advance_while(|c| c.is_alphanumeric() || c == '_' || c == '\'');
        let text = &self.src[start..self.pos];
        let sym = Symbol::intern(text);
        Token {
            kind: TokenKind::ConId(sym),
        }
    }

    /// Lex a number literal.
    fn lex_number(&mut self, start: usize) -> Token {
        // Check for hex/octal/binary
        if self.peek() == Some('0') {
            match self.peek2() {
                Some('x') | Some('X') => {
                    self.advance();
                    self.advance();
                    self.advance_while(|c| c.is_ascii_hexdigit() || c == '_');
                    let text = &self.src[start..self.pos];
                    return Token {
                        kind: TokenKind::IntLit(text.to_string()),
                    };
                }
                Some('o') | Some('O') => {
                    self.advance();
                    self.advance();
                    self.advance_while(|c| matches!(c, '0'..='7') || c == '_');
                    let text = &self.src[start..self.pos];
                    return Token {
                        kind: TokenKind::IntLit(text.to_string()),
                    };
                }
                Some('b') | Some('B') => {
                    self.advance();
                    self.advance();
                    self.advance_while(|c| c == '0' || c == '1' || c == '_');
                    let text = &self.src[start..self.pos];
                    return Token {
                        kind: TokenKind::IntLit(text.to_string()),
                    };
                }
                _ => {}
            }
        }

        // Decimal integer part
        self.advance_while(|c| c.is_ascii_digit() || c == '_');

        // Check for float
        if self.peek() == Some('.') && self.peek2().is_some_and(|c| c.is_ascii_digit()) {
            self.advance(); // .
            self.advance_while(|c| c.is_ascii_digit() || c == '_');

            // Exponent
            if let Some('e') | Some('E') = self.peek() {
                self.advance();
                if let Some('+') | Some('-') = self.peek() {
                    self.advance();
                }
                self.advance_while(|c| c.is_ascii_digit() || c == '_');
            }

            let text = &self.src[start..self.pos];
            return Token {
                kind: TokenKind::FloatLit(text.to_string()),
            };
        }

        let text = &self.src[start..self.pos];
        Token {
            kind: TokenKind::IntLit(text.to_string()),
        }
    }

    /// Lex a string literal.
    fn lex_string(&mut self, _start: usize) -> Token {
        self.advance(); // Opening quote
        let content_start = self.pos;

        loop {
            match self.peek() {
                Some('"') => {
                    let content = &self.src[content_start..self.pos];
                    self.advance(); // Closing quote
                    return Token {
                        kind: TokenKind::StringLit(content.to_string()),
                    };
                }
                Some('\\') => {
                    self.advance();
                    self.advance(); // Escaped character
                }
                Some(_) => {
                    self.advance();
                }
                None => {
                    // Unterminated string
                    let content = &self.src[content_start..self.pos];
                    return Token {
                        kind: TokenKind::StringLit(content.to_string()),
                    };
                }
            }
        }
    }

    /// Lex a character literal.
    fn lex_char(&mut self, _start: usize) -> Token {
        self.advance(); // Opening quote

        let c = match self.peek() {
            Some('\\') => {
                self.advance();
                match self.advance() {
                    Some('n') => '\n',
                    Some('t') => '\t',
                    Some('r') => '\r',
                    Some('\\') => '\\',
                    Some('\'') => '\'',
                    Some('0') => '\0',
                    Some(c) => c,
                    None => '\0',
                }
            }
            Some(c) => {
                self.advance();
                c
            }
            None => '\0',
        };

        if self.peek() == Some('\'') {
            self.advance(); // Closing quote
        }

        Token {
            kind: TokenKind::CharLit(c),
        }
    }

    /// Lex an operator.
    fn lex_operator(&mut self, start: usize) -> Token {
        self.advance_while(|c| is_operator_char(c));
        let text = &self.src[start..self.pos];

        let kind = match text {
            "=" => TokenKind::Eq,
            "|" => TokenKind::Pipe,
            "\\" => TokenKind::Backslash,
            "->" => TokenKind::Arrow,
            "<-" => TokenKind::LeftArrow,
            "=>" => TokenKind::FatArrow,
            "::" => TokenKind::DoubleColon,
            ".." => TokenKind::DotDot,
            "@" => TokenKind::At,
            "~" => TokenKind::Tilde,
            _ => TokenKind::Operator(Symbol::intern(text)),
        };

        Token { kind }
    }

    /// Lex the next token.
    fn lex_token(&mut self) -> Option<Spanned<Token>> {
        self.skip_whitespace();

        let start = self.pos;
        let c = self.peek()?;

        let token = match c {
            // Identifiers and keywords
            c if c.is_lowercase() || c == '_' => self.lex_ident(start),

            // Constructors
            c if c.is_uppercase() => self.lex_conid(start),

            // Numbers
            c if c.is_ascii_digit() => self.lex_number(start),

            // Strings
            '"' => self.lex_string(start),

            // Characters
            '\'' => self.lex_char(start),

            // Single-character tokens
            '(' => {
                self.advance();
                Token {
                    kind: TokenKind::LParen,
                }
            }
            ')' => {
                self.advance();
                Token {
                    kind: TokenKind::RParen,
                }
            }
            '[' => {
                self.advance();
                Token {
                    kind: TokenKind::LBracket,
                }
            }
            ']' => {
                self.advance();
                Token {
                    kind: TokenKind::RBracket,
                }
            }
            '{' => {
                self.advance();
                Token {
                    kind: TokenKind::LBrace,
                }
            }
            '}' => {
                self.advance();
                Token {
                    kind: TokenKind::RBrace,
                }
            }
            ',' => {
                self.advance();
                Token {
                    kind: TokenKind::Comma,
                }
            }
            ';' => {
                self.advance();
                Token {
                    kind: TokenKind::Semi,
                }
            }
            '`' => {
                self.advance();
                Token {
                    kind: TokenKind::Backtick,
                }
            }

            // Operators
            c if is_operator_char(c) => self.lex_operator(start),

            // Unknown
            _ => {
                self.advance();
                Token {
                    kind: TokenKind::Error,
                }
            }
        };

        let span = Span::from_raw(start as u32, self.pos as u32);
        Some(Spanned::new(token, span))
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Spanned<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        // Return pending tokens first (from layout rule)
        if let Some(tok) = self.pending.pop() {
            return Some(tok);
        }

        self.lex_token()
    }
}

/// Check if a character can be part of an operator.
fn is_operator_char(c: char) -> bool {
    matches!(
        c,
        '!' | '#'
            | '$'
            | '%'
            | '&'
            | '*'
            | '+'
            | '.'
            | '/'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '\\'
            | '^'
            | '|'
            | '-'
            | '~'
            | ':'
    )
}

/// Lex source code into a vector of tokens.
#[must_use]
pub fn lex(src: &str) -> Vec<Spanned<Token>> {
    Lexer::new(src).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_keywords() {
        let tokens = lex("let in where if then else");
        let kinds: Vec<_> = tokens.iter().map(|t| &t.node.kind).collect();

        assert_eq!(
            kinds,
            vec![
                &TokenKind::Let,
                &TokenKind::In,
                &TokenKind::Where,
                &TokenKind::If,
                &TokenKind::Then,
                &TokenKind::Else,
            ]
        );
    }

    #[test]
    fn test_lex_numbers() {
        let tokens = lex("42 3.14 0xFF 0b1010");
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn test_lex_operators() {
        let tokens = lex("+ - * / -> <-");
        assert_eq!(tokens.len(), 6);
    }

    #[test]
    fn test_lex_string() {
        let tokens = lex(r#""hello world""#);
        assert_eq!(tokens.len(), 1);
        match &tokens[0].node.kind {
            TokenKind::StringLit(s) => assert_eq!(s, "hello world"),
            _ => panic!("expected string literal"),
        }
    }
}
