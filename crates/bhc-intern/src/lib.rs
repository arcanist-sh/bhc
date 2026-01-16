//! String interning for efficient symbol handling.
//!
//! This crate provides interned strings (symbols) that enable O(1) equality
//! comparisons and reduced memory usage for repeated strings.

#![warn(missing_docs)]

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::LazyLock;

/// The global interner for symbols.
static INTERNER: LazyLock<Interner> = LazyLock::new(Interner::new);

/// An interned string symbol.
///
/// Symbols are cheap to copy and compare (O(1) equality).
/// The actual string data is stored in a global interner.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Symbol(u32);

impl Symbol {
    /// Intern a string and return its symbol.
    #[must_use]
    pub fn intern(s: &str) -> Self {
        INTERNER.intern(s)
    }

    /// Get the string value of this symbol.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        INTERNER.get(self)
    }

    /// Get the raw index of this symbol.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Create a symbol from a raw index.
    ///
    /// # Safety
    ///
    /// The index must have been obtained from a valid symbol.
    #[must_use]
    pub const unsafe fn from_raw(idx: u32) -> Self {
        Self(idx)
    }

    /// Check if this symbol is empty.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.as_str().is_empty()
    }

    /// Get the length of the symbol's string.
    #[must_use]
    pub fn len(self) -> usize {
        self.as_str().len()
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol({:?})", self.as_str())
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Self::intern(s)
    }
}

impl From<String> for Symbol {
    fn from(s: String) -> Self {
        Self::intern(&s)
    }
}

impl AsRef<str> for Symbol {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<str> for Symbol {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Symbol {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

/// The string interner that stores all interned strings.
struct Interner {
    map: RwLock<FxHashMap<&'static str, Symbol>>,
    strings: RwLock<Vec<&'static str>>,
}

impl Interner {
    fn new() -> Self {
        Self {
            map: RwLock::new(FxHashMap::default()),
            strings: RwLock::new(Vec::new()),
        }
    }

    fn intern(&self, s: &str) -> Symbol {
        // Fast path: check if already interned
        {
            let map = self.map.read();
            if let Some(&sym) = map.get(s) {
                return sym;
            }
        }

        // Slow path: intern the string
        let mut map = self.map.write();
        let mut strings = self.strings.write();

        // Double-check after acquiring write lock
        if let Some(&sym) = map.get(s) {
            return sym;
        }

        // Leak the string to get a static lifetime
        let interned: &'static str = Box::leak(s.to_string().into_boxed_str());
        let sym = Symbol(strings.len() as u32);

        strings.push(interned);
        map.insert(interned, sym);

        sym
    }

    fn get(&self, sym: Symbol) -> &'static str {
        let strings = self.strings.read();
        strings[sym.0 as usize]
    }
}

/// Pre-interned symbols for common identifiers.
pub mod kw {
    use super::Symbol;
    use std::sync::LazyLock;

    macro_rules! define_keywords {
        ($($name:ident => $string:literal),* $(,)?) => {
            $(
                #[doc = concat!("The `", $string, "` keyword.")]
                pub static $name: LazyLock<Symbol> = LazyLock::new(|| Symbol::intern($string));
            )*

            /// Intern all keywords. Call this at startup for better performance.
            pub fn intern_all() {
                $(
                    let _ = *$name;
                )*
            }
        };
    }

    define_keywords! {
        // Haskell keywords
        CASE => "case",
        CLASS => "class",
        DATA => "data",
        DEFAULT => "default",
        DERIVING => "deriving",
        DO => "do",
        ELSE => "else",
        FORALL => "forall",
        FOREIGN => "foreign",
        IF => "if",
        IMPORT => "import",
        IN => "in",
        INFIX => "infix",
        INFIXL => "infixl",
        INFIXR => "infixr",
        INSTANCE => "instance",
        LET => "let",
        MODULE => "module",
        NEWTYPE => "newtype",
        OF => "of",
        QUALIFIED => "qualified",
        THEN => "then",
        TYPE => "type",
        WHERE => "where",

        // BHC/H26 extensions
        LAZY => "lazy",
        STRICT => "strict",
        PROFILE => "profile",
        EDITION => "edition",

        // Common type names
        INT => "Int",
        FLOAT => "Float",
        DOUBLE => "Double",
        BOOL => "Bool",
        CHAR => "Char",
        STRING => "String",
        UNIT => "()",

        // Common constructors
        TRUE => "True",
        FALSE => "False",
        JUST => "Just",
        NOTHING => "Nothing",
        LEFT => "Left",
        RIGHT => "Right",

        // Underscore
        UNDERSCORE => "_",
    }
}

/// An identifier with a name symbol.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ident {
    /// The symbol for this identifier's name.
    pub name: Symbol,
}

impl Ident {
    /// Create a new identifier.
    #[must_use]
    pub fn new(name: Symbol) -> Self {
        Self { name }
    }

    /// Create an identifier from a string.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        Self {
            name: Symbol::intern(s),
        }
    }

    /// Get the string value of this identifier.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        self.name.as_str()
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ident({:?})", self.name.as_str())
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name.as_str())
    }
}

impl From<&str> for Ident {
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}

impl From<Symbol> for Ident {
    fn from(name: Symbol) -> Self {
        Self::new(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_interning() {
        let s1 = Symbol::intern("hello");
        let s2 = Symbol::intern("hello");
        let s3 = Symbol::intern("world");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_eq!(s1.as_str(), "hello");
    }

    #[test]
    fn test_symbol_comparison() {
        let s1 = Symbol::intern("apple");
        let s2 = Symbol::intern("banana");

        assert!(s1 < s2);
        assert_eq!(s1, "apple");
    }

    #[test]
    fn test_ident() {
        let id = Ident::from_str("foo");
        assert_eq!(id.as_str(), "foo");
    }
}
