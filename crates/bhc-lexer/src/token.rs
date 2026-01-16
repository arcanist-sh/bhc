//! Token definitions for the BHC lexer.

use bhc_intern::Symbol;

/// A token produced by the lexer.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    /// The kind of token.
    pub kind: TokenKind,
}

/// The kind of token.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // Keywords
    /// `case`
    Case,
    /// `class`
    Class,
    /// `data`
    Data,
    /// `default`
    Default,
    /// `deriving`
    Deriving,
    /// `do`
    Do,
    /// `else`
    Else,
    /// `forall`
    Forall,
    /// `foreign`
    Foreign,
    /// `if`
    If,
    /// `import`
    Import,
    /// `in`
    In,
    /// `infix`
    Infix,
    /// `infixl`
    Infixl,
    /// `infixr`
    Infixr,
    /// `instance`
    Instance,
    /// `let`
    Let,
    /// `module`
    Module,
    /// `newtype`
    Newtype,
    /// `of`
    Of,
    /// `qualified`
    Qualified,
    /// `then`
    Then,
    /// `type`
    Type,
    /// `where`
    Where,

    // H26 extensions
    /// `lazy`
    Lazy,
    /// `strict`
    Strict,

    // Identifiers
    /// A lowercase identifier.
    Ident(Symbol),
    /// An uppercase identifier (constructor/type name).
    ConId(Symbol),
    /// An operator symbol.
    Operator(Symbol),

    // Literals
    /// An integer literal.
    IntLit(String),
    /// A floating-point literal.
    FloatLit(String),
    /// A character literal.
    CharLit(char),
    /// A string literal.
    StringLit(String),

    // Punctuation
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `,`
    Comma,
    /// `;`
    Semi,
    /// `` ` ``
    Backtick,
    /// `_`
    Underscore,

    // Operators
    /// `=`
    Eq,
    /// `|`
    Pipe,
    /// `\`
    Backslash,
    /// `->`
    Arrow,
    /// `<-`
    LeftArrow,
    /// `=>`
    FatArrow,
    /// `::`
    DoubleColon,
    /// `..`
    DotDot,
    /// `@`
    At,
    /// `~`
    Tilde,

    // Layout tokens (inserted by layout rule)
    /// Virtual `{` from layout rule.
    VirtualLBrace,
    /// Virtual `}` from layout rule.
    VirtualRBrace,
    /// Virtual `;` from layout rule.
    VirtualSemi,

    // Special
    /// End of file.
    Eof,
    /// Lexer error.
    Error,
}

impl TokenKind {
    /// Check if this is a keyword.
    #[must_use]
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Self::Case
                | Self::Class
                | Self::Data
                | Self::Default
                | Self::Deriving
                | Self::Do
                | Self::Else
                | Self::Forall
                | Self::Foreign
                | Self::If
                | Self::Import
                | Self::In
                | Self::Infix
                | Self::Infixl
                | Self::Infixr
                | Self::Instance
                | Self::Let
                | Self::Module
                | Self::Newtype
                | Self::Of
                | Self::Qualified
                | Self::Then
                | Self::Type
                | Self::Where
                | Self::Lazy
                | Self::Strict
        )
    }

    /// Check if this is a literal.
    #[must_use]
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            Self::IntLit(_) | Self::FloatLit(_) | Self::CharLit(_) | Self::StringLit(_)
        )
    }

    /// Check if this token starts a layout block.
    #[must_use]
    pub fn starts_layout(&self) -> bool {
        matches!(self, Self::Where | Self::Let | Self::Do | Self::Of)
    }

    /// Get the name of the token for error messages.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Case => "`case`",
            Self::Class => "`class`",
            Self::Data => "`data`",
            Self::Default => "`default`",
            Self::Deriving => "`deriving`",
            Self::Do => "`do`",
            Self::Else => "`else`",
            Self::Forall => "`forall`",
            Self::Foreign => "`foreign`",
            Self::If => "`if`",
            Self::Import => "`import`",
            Self::In => "`in`",
            Self::Infix => "`infix`",
            Self::Infixl => "`infixl`",
            Self::Infixr => "`infixr`",
            Self::Instance => "`instance`",
            Self::Let => "`let`",
            Self::Module => "`module`",
            Self::Newtype => "`newtype`",
            Self::Of => "`of`",
            Self::Qualified => "`qualified`",
            Self::Then => "`then`",
            Self::Type => "`type`",
            Self::Where => "`where`",
            Self::Lazy => "`lazy`",
            Self::Strict => "`strict`",
            Self::Ident(_) => "identifier",
            Self::ConId(_) => "constructor",
            Self::Operator(_) => "operator",
            Self::IntLit(_) => "integer literal",
            Self::FloatLit(_) => "float literal",
            Self::CharLit(_) => "character literal",
            Self::StringLit(_) => "string literal",
            Self::LParen => "`(`",
            Self::RParen => "`)`",
            Self::LBracket => "`[`",
            Self::RBracket => "`]`",
            Self::LBrace => "`{`",
            Self::RBrace => "`}`",
            Self::Comma => "`,`",
            Self::Semi => "`;`",
            Self::Backtick => "`` ` ``",
            Self::Underscore => "`_`",
            Self::Eq => "`=`",
            Self::Pipe => "`|`",
            Self::Backslash => "`\\`",
            Self::Arrow => "`->`",
            Self::LeftArrow => "`<-`",
            Self::FatArrow => "`=>`",
            Self::DoubleColon => "`::`",
            Self::DotDot => "`..`",
            Self::At => "`@`",
            Self::Tilde => "`~`",
            Self::VirtualLBrace => "virtual `{`",
            Self::VirtualRBrace => "virtual `}`",
            Self::VirtualSemi => "virtual `;`",
            Self::Eof => "end of file",
            Self::Error => "error",
        }
    }
}
