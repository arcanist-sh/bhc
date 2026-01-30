//! Runtime values for the Core IR interpreter.
//!
//! This module defines the `Value` type that represents values during
//! interpretation of Core IR expressions.

use std::fmt;
use std::sync::Arc;

use bhc_intern::Symbol;

use crate::uarray::UArray;
use crate::{DataCon, Expr, Var};

/// A runtime value produced by evaluating Core IR.
#[derive(Clone)]
pub enum Value {
    /// An integer value.
    Int(i64),

    /// An arbitrary precision integer.
    Integer(i128),

    /// A single-precision float.
    Float(f32),

    /// A double-precision float.
    Double(f64),

    /// A character.
    Char(char),

    /// A string.
    String(Arc<str>),

    /// A closure (lambda with captured environment).
    Closure(Closure),

    /// A data constructor value (fully or partially applied).
    Data(DataValue),

    /// A thunk (unevaluated expression with environment).
    /// Used for lazy evaluation in Default Profile.
    Thunk(Thunk),

    /// A special value representing a primitive operation.
    PrimOp(PrimOp),

    /// A partially applied primitive operation.
    PartialPrimOp(PrimOp, Vec<Value>),

    /// An unboxed integer array.
    UArrayInt(UArray<i64>),

    /// An unboxed double array.
    UArrayDouble(UArray<f64>),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{n}"),
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}f"),
            Self::Double(n) => write!(f, "{n}"),
            Self::Char(c) => write!(f, "{c:?}"),
            Self::String(s) => write!(f, "{s:?}"),
            Self::Closure(c) => write!(f, "<closure {}>", c.var.name),
            Self::Data(d) => {
                write!(f, "{}", d.con.name)?;
                for arg in &d.args {
                    write!(f, " {arg:?}")?;
                }
                Ok(())
            }
            Self::Thunk(_) => write!(f, "<thunk>"),
            Self::PrimOp(op) => write!(f, "<primop {op:?}>"),
            Self::PartialPrimOp(op, args) => {
                write!(f, "<partial {op:?} applied to {} args>", args.len())
            }
            Self::UArrayInt(arr) => write!(f, "UArray[Int; {}]", arr.len()),
            Self::UArrayDouble(arr) => write!(f, "UArray[Double; {}]", arr.len()),
        }
    }
}

impl Value {
    /// Returns true if this value needs to be forced (is a thunk).
    #[must_use]
    pub fn is_thunk(&self) -> bool {
        matches!(self, Self::Thunk(_))
    }

    /// Converts an integer value, returning None if not an integer.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(n) => Some(*n),
            Self::Integer(n) => i64::try_from(*n).ok(),
            _ => None,
        }
    }

    /// Converts to a double value, returning None if not numeric.
    #[must_use]
    pub fn as_double(&self) -> Option<f64> {
        match self {
            Self::Double(n) => Some(*n),
            Self::Float(n) => Some(f64::from(*n)),
            Self::Int(n) => Some(*n as f64),
            Self::Integer(n) => Some(*n as f64),
            _ => None,
        }
    }

    /// Converts to a bool value (data constructor True/False).
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Data(d) if d.args.is_empty() => {
                let name = d.con.name.as_str();
                match name {
                    "True" => Some(true),
                    "False" => Some(false),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Creates a boolean value.
    #[must_use]
    pub fn bool(b: bool) -> Self {
        use bhc_types::{Kind, TyCon};
        let name = if b {
            Symbol::intern("True")
        } else {
            Symbol::intern("False")
        };
        Self::Data(DataValue {
            con: DataCon {
                name,
                ty_con: TyCon::new(Symbol::intern("Bool"), Kind::Star),
                tag: if b { 1 } else { 0 },
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Creates a unit value `()`.
    #[must_use]
    pub fn unit() -> Self {
        use bhc_types::{Kind, TyCon};
        Self::Data(DataValue {
            con: DataCon {
                name: Symbol::intern("()"),
                ty_con: TyCon::new(Symbol::intern("()"), Kind::Star),
                tag: 0,
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Creates an empty list `[]`.
    #[must_use]
    pub fn nil() -> Self {
        use bhc_types::{Kind, TyCon};
        Self::Data(DataValue {
            con: DataCon {
                name: Symbol::intern("[]"),
                ty_con: TyCon::new(Symbol::intern("[]"), Kind::star_to_star()),
                tag: 0,
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Creates a cons cell `x : xs`.
    #[must_use]
    pub fn cons(head: Value, tail: Value) -> Self {
        use bhc_types::{Kind, TyCon};
        Self::Data(DataValue {
            con: DataCon {
                name: Symbol::intern(":"),
                ty_con: TyCon::new(Symbol::intern("[]"), Kind::star_to_star()),
                tag: 1,
                arity: 2,
            },
            args: vec![head, tail],
        })
    }

    /// Converts a list value to a Vec, returning None if not a list.
    #[must_use]
    pub fn as_list(&self) -> Option<Vec<Value>> {
        let mut result = Vec::new();
        let mut current = self;

        loop {
            match current {
                Self::Data(d) if d.con.name.as_str() == "[]" => {
                    return Some(result);
                }
                Self::Data(d) if d.con.name.as_str() == ":" && d.args.len() == 2 => {
                    result.push(d.args[0].clone());
                    current = &d.args[1];
                }
                _ => return None,
            }
        }
    }

    /// Creates a list value from a Vec.
    #[must_use]
    pub fn from_list(values: Vec<Value>) -> Self {
        values
            .into_iter()
            .rev()
            .fold(Self::nil(), |acc, v| Self::cons(v, acc))
    }

    /// Creates an integer UArray from a list of int values.
    #[must_use]
    pub fn uarray_int_from_list(list: &Self) -> Option<Self> {
        let values = list.as_list()?;
        let ints: Option<Vec<i64>> = values.iter().map(Self::as_int).collect();
        Some(Self::UArrayInt(UArray::from_vec(ints?)))
    }

    /// Creates a double UArray from a list of double values.
    #[must_use]
    pub fn uarray_double_from_list(list: &Self) -> Option<Self> {
        let values = list.as_list()?;
        let doubles: Option<Vec<f64>> = values.iter().map(Self::as_double).collect();
        Some(Self::UArrayDouble(UArray::from_vec(doubles?)))
    }

    /// Converts a UArray to a list value.
    #[must_use]
    pub fn uarray_to_list(&self) -> Option<Self> {
        match self {
            Self::UArrayInt(arr) => {
                let values: Vec<Self> = arr.to_vec().into_iter().map(Self::Int).collect();
                Some(Self::from_list(values))
            }
            Self::UArrayDouble(arr) => {
                let values: Vec<Self> = arr.to_vec().into_iter().map(Self::Double).collect();
                Some(Self::from_list(values))
            }
            _ => None,
        }
    }

    /// Returns the UArray as an integer array, if applicable.
    #[must_use]
    pub fn as_uarray_int(&self) -> Option<&UArray<i64>> {
        match self {
            Self::UArrayInt(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns the UArray as a double array, if applicable.
    #[must_use]
    pub fn as_uarray_double(&self) -> Option<&UArray<f64>> {
        match self {
            Self::UArrayDouble(arr) => Some(arr),
            _ => None,
        }
    }
}

/// A closure capturing a lambda and its environment.
#[derive(Clone)]
pub struct Closure {
    /// The bound variable.
    pub var: Var,
    /// The body expression.
    pub body: Box<Expr>,
    /// The captured environment.
    pub env: super::Env,
}

/// A data constructor value with its arguments.
#[derive(Clone, Debug)]
pub struct DataValue {
    /// The data constructor.
    pub con: DataCon,
    /// The constructor arguments (may be partial).
    pub args: Vec<Value>,
}

impl DataValue {
    /// Returns true if this data value is fully applied.
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        self.args.len() == self.con.arity as usize
    }
}

/// A thunk representing an unevaluated expression.
#[derive(Clone)]
pub struct Thunk {
    /// The unevaluated expression.
    pub expr: Box<Expr>,
    /// The environment at thunk creation time.
    pub env: super::Env,
}

/// Primitive operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimOp {
    // Arithmetic
    /// Integer addition.
    AddInt,
    /// Integer subtraction.
    SubInt,
    /// Integer multiplication.
    MulInt,
    /// Integer division.
    DivInt,
    /// Integer modulo.
    ModInt,
    /// Integer negation.
    NegInt,

    // Floating point
    /// Double addition.
    AddDouble,
    /// Double subtraction.
    SubDouble,
    /// Double multiplication.
    MulDouble,
    /// Double division.
    DivDouble,
    /// Double negation.
    NegDouble,

    // Comparison
    /// Integer equality.
    EqInt,
    /// Integer less-than.
    LtInt,
    /// Integer less-than-or-equal.
    LeInt,
    /// Integer greater-than.
    GtInt,
    /// Integer greater-than-or-equal.
    GeInt,

    /// Double equality.
    EqDouble,
    /// Double less-than.
    LtDouble,

    // Boolean
    /// Boolean and.
    AndBool,
    /// Boolean or.
    OrBool,
    /// Boolean not.
    NotBool,

    // Conversion
    /// Int to Double.
    IntToDouble,
    /// Double to Int.
    DoubleToInt,

    // Char/String
    /// Character equality.
    EqChar,
    /// Char to Int (ord).
    CharToInt,
    /// Int to Char (chr).
    IntToChar,

    // Seq (for strict evaluation)
    /// Evaluate first arg to WHNF, return second.
    Seq,

    // Error
    /// Throw an error.
    Error,

    // UArray operations
    /// Create an integer UArray from a list.
    UArrayFromList,
    /// Convert a UArray back to a list.
    UArrayToList,
    /// Map a function over a UArray.
    UArrayMap,
    /// Zip two UArrays with a function.
    UArrayZipWith,
    /// Fold over a UArray.
    UArrayFold,
    /// Sum all elements in a UArray.
    UArraySum,
    /// Get the length of a UArray.
    UArrayLength,
    /// Create a range [start..end).
    UArrayRange,

    // List operations
    /// Concatenate two lists.
    Concat,
    /// Map a function over a list and concatenate results.
    ConcatMap,
    /// Append an element to a list.
    Append,

    // Monad operations (for list monad)
    /// Monadic bind (>>=) for lists: xs >>= f = concatMap f xs
    ListBind,
    /// Monadic then (>>) for lists: xs >> ys = xs >>= \_ -> ys
    ListThen,
    /// Monadic return for lists: return x = [x]
    ListReturn,

    // Additional list operations
    /// Right fold: foldr f z xs
    Foldr,
    /// Left fold: foldl f z xs
    Foldl,
    /// Strict left fold: foldl' f z xs
    FoldlStrict,
    /// Filter: filter p xs
    Filter,
    /// Zip two lists: zip xs ys
    Zip,
    /// Zip with function: zipWith f xs ys
    ZipWith,
    /// Take n elements: take n xs
    Take,
    /// Drop n elements: drop n xs
    Drop,
    /// Head of list: head xs
    Head,
    /// Tail of list: tail xs
    Tail,
    /// Last element: last xs
    Last,
    /// All but last: init xs
    Init,
    /// Reverse a list: reverse xs
    Reverse,
    /// Null check: null xs
    Null,
    /// Element at index: xs !! n
    Index,
    /// Replicate: replicate n x
    Replicate,
    /// Enumeration: enumFromTo start end
    EnumFromTo,

    // Additional list operations (second batch)
    /// Even predicate: even n
    Even,
    /// Odd predicate: odd n
    Odd,
    /// List membership: elem x xs
    Elem,
    /// List non-membership: notElem x xs
    NotElem,
    /// Take while predicate holds: takeWhile p xs
    TakeWhile,
    /// Drop while predicate holds: dropWhile p xs
    DropWhile,
    /// Split at predicate: span p xs
    Span,
    /// Split at predicate negation: break p xs
    Break,
    /// Split at index: splitAt n xs
    SplitAt,
    /// Iterate function: iterate f x (returns first 1000 elements)
    Iterate,
    /// Repeat value: repeat x (returns first 1000 elements)
    Repeat,
    /// Cycle list: cycle xs (returns first 1000 elements)
    Cycle,
    /// Lookup in assoc list: lookup k xs
    Lookup,
    /// Unzip pairs: unzip xs
    Unzip,
    /// Product of list: product xs
    Product,
    /// Flip function arguments: flip f x y = f y x
    Flip,
    /// Minimum of two: min a b
    Min,
    /// Maximum of two: max a b
    Max,
    /// Identity conversion: fromIntegral n
    FromIntegral,
    /// Maybe eliminator: maybe def f m
    MaybeElim,
    /// Default from Maybe: fromMaybe def m
    FromMaybe,
    /// Either eliminator: either f g e
    EitherElim,
    /// isJust :: Maybe a -> Bool
    IsJust,
    /// isNothing :: Maybe a -> Bool
    IsNothing,
    /// Absolute value: abs n
    Abs,
    /// Sign: signum n
    Signum,
    /// curry :: ((a, b) -> c) -> a -> b -> c
    Curry,
    /// uncurry :: (a -> b -> c) -> (a, b) -> c
    Uncurry,
    /// swap :: (a, b) -> (b, a)
    Swap,
    /// any :: (a -> Bool) -> [a] -> Bool
    Any,
    /// all :: (a -> Bool) -> [a] -> Bool
    All,
    /// and :: [Bool] -> Bool
    And,
    /// or :: [Bool] -> Bool
    Or,
    /// lines :: String -> [String]
    Lines,
    /// unlines :: [String] -> String
    Unlines,
    /// words :: String -> [String]
    Words,
    /// unwords :: [String] -> String
    Unwords,
    /// show :: a -> String
    Show,
    /// id :: a -> a
    Id,
    /// const :: a -> b -> a
    Const,

    // IO operations
    /// Print a string followed by newline.
    PutStrLn,
    /// Print a string without newline.
    PutStr,
    /// Print a value using Show (for now, uses Debug).
    Print,
    /// Read a line from stdin.
    GetLine,
    /// IO bind (>>=) for IO monad.
    IoBind,
    /// IO then (>>) for IO monad.
    IoThen,
    /// IO return/pure.
    IoReturn,

    // Polymorphic monad operations (dispatch based on first argument type)
    /// Polymorphic bind (>>=): dispatches to IoBind or ListBind based on first arg.
    MonadBind,
    /// Polymorphic then (>>): dispatches to IoThen or ListThen based on first arg.
    MonadThen,

    // Prelude: Enum operations
    /// succ :: Enum a => a -> a
    Succ,
    /// pred :: Enum a => a -> a
    Pred,
    /// toEnum :: Enum a => Int -> a
    ToEnum,
    /// fromEnum :: Enum a => a -> Int
    FromEnum,

    // Prelude: Integral operations
    /// gcd :: Integral a => a -> a -> a
    Gcd,
    /// lcm :: Integral a => a -> a -> a
    Lcm,
    /// quot :: Integral a => a -> a -> a
    Quot,
    /// rem :: Integral a => a -> a -> a
    Rem,
    /// quotRem :: Integral a => a -> a -> (a, a)
    QuotRem,
    /// divMod :: Integral a => a -> a -> (a, a)
    DivMod,
    /// subtract :: Num a => a -> a -> a
    Subtract,

    // Prelude: Scan operations
    /// scanl :: (b -> a -> b) -> b -> [a] -> [b]
    Scanl,
    /// scanr :: (a -> b -> b) -> b -> [a] -> [b]
    Scanr,
    /// scanl1 :: (a -> a -> a) -> [a] -> [a]
    Scanl1,
    /// scanr1 :: (a -> a -> a) -> [a] -> [a]
    Scanr1,

    // Prelude: More list operations
    /// maximum :: Ord a => [a] -> a
    Maximum,
    /// minimum :: Ord a => [a] -> a
    Minimum,
    /// zip3 :: [a] -> [b] -> [c] -> [(a,b,c)]
    Zip3,
    /// zipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
    ZipWith3,
    /// unzip3 :: [(a,b,c)] -> ([a],[b],[c])
    Unzip3,

    // Prelude: Show helpers
    /// showString :: String -> ShowS
    ShowString,
    /// showChar :: Char -> ShowS
    ShowChar,
    /// showParen :: Bool -> ShowS -> ShowS
    ShowParen,

    // Prelude: IO operations
    /// getChar :: IO Char
    GetChar,
    /// getContents :: IO String
    GetContents,
    /// readFile :: FilePath -> IO String
    ReadFile,
    /// writeFile :: FilePath -> String -> IO ()
    WriteFile,
    /// appendFile :: FilePath -> String -> IO ()
    AppendFile,
    /// interact :: (String -> String) -> IO ()
    Interact,

    // Prelude: otherwise and misc
    /// otherwise :: Bool (always True)
    Otherwise,
    /// until :: (a -> Bool) -> (a -> a) -> a -> a
    Until,
    /// asTypeOf :: a -> a -> a
    AsTypeOf,
    /// realToFrac :: (Real a, Fractional b) => a -> b
    RealToFrac,

    // Data.List operations
    /// sort :: Ord a => [a] -> [a]
    Sort,
    /// sortBy :: (a -> a -> Ordering) -> [a] -> [a]
    SortBy,
    /// sortOn :: Ord b => (a -> b) -> [a] -> [a]
    SortOn,
    /// nub :: Eq a => [a] -> [a]
    Nub,
    /// nubBy :: (a -> a -> Bool) -> [a] -> [a]
    NubBy,
    /// group :: Eq a => [a] -> [[a]]
    Group,
    /// groupBy :: (a -> a -> Bool) -> [a] -> [[a]]
    GroupBy,
    /// intersperse :: a -> [a] -> [a]
    Intersperse,
    /// intercalate :: [a] -> [[a]] -> [a]
    Intercalate,
    /// transpose :: [[a]] -> [[a]]
    Transpose,
    /// subsequences :: [a] -> [[a]]
    Subsequences,
    /// permutations :: [a] -> [[a]]
    Permutations,
    /// partition :: (a -> Bool) -> [a] -> ([a], [a])
    Partition,
    /// find :: (a -> Bool) -> [a] -> Maybe a
    Find,
    /// stripPrefix :: Eq a => [a] -> [a] -> Maybe [a]
    StripPrefix,
    /// isPrefixOf :: Eq a => [a] -> [a] -> Bool
    IsPrefixOf,
    /// isSuffixOf :: Eq a => [a] -> [a] -> Bool
    IsSuffixOf,
    /// isInfixOf :: Eq a => [a] -> [a] -> Bool
    IsInfixOf,
    /// delete :: Eq a => a -> [a] -> [a]
    Delete,
    /// deleteBy :: (a -> a -> Bool) -> a -> [a] -> [a]
    DeleteBy,
    /// union :: Eq a => [a] -> [a] -> [a]
    Union,
    /// unionBy :: (a -> a -> Bool) -> [a] -> [a] -> [a]
    UnionBy,
    /// intersect :: Eq a => [a] -> [a] -> [a]
    Intersect,
    /// intersectBy :: (a -> a -> Bool) -> [a] -> [a] -> [a]
    IntersectBy,
    /// (\\) :: Eq a => [a] -> [a] -> [a]
    ListDiff,
    /// tails :: [a] -> [[a]]
    Tails,
    /// inits :: [a] -> [[a]]
    Inits,
    /// mapAccumL :: (acc -> x -> (acc, y)) -> acc -> [x] -> (acc, [y])
    MapAccumL,
    /// mapAccumR :: (acc -> x -> (acc, y)) -> acc -> [x] -> (acc, [y])
    MapAccumR,
    /// unfoldr :: (b -> Maybe (a, b)) -> b -> [a]
    Unfoldr,
    /// genericLength :: Num i => [a] -> i
    GenericLength,
    /// genericTake :: Integral i => i -> [a] -> [a]
    GenericTake,
    /// genericDrop :: Integral i => i -> [a] -> [a]
    GenericDrop,

    // Data.Char operations
    /// isAlpha :: Char -> Bool
    IsAlpha,
    /// isAlphaNum :: Char -> Bool
    IsAlphaNum,
    /// isAscii :: Char -> Bool
    IsAscii,
    /// isControl :: Char -> Bool
    IsControl,
    /// isDigit :: Char -> Bool
    IsDigit,
    /// isHexDigit :: Char -> Bool
    IsHexDigit,
    /// isLetter :: Char -> Bool
    IsLetter,
    /// isLower :: Char -> Bool
    IsLower,
    /// isNumber :: Char -> Bool
    IsNumber,
    /// isPrint :: Char -> Bool
    IsPrint,
    /// isPunctuation :: Char -> Bool
    IsPunctuation,
    /// isSpace :: Char -> Bool
    IsSpace,
    /// isSymbol :: Char -> Bool
    IsSymbol,
    /// isUpper :: Char -> Bool
    IsUpper,
    /// toLower :: Char -> Char
    ToLower,
    /// toUpper :: Char -> Char
    ToUpper,
    /// toTitle :: Char -> Char
    ToTitle,
    /// digitToInt :: Char -> Int
    DigitToInt,
    /// intToDigit :: Int -> Char
    IntToDigit,
    /// isLatin1 :: Char -> Bool
    IsLatin1,
    /// isAsciiLower :: Char -> Bool
    IsAsciiLower,
    /// isAsciiUpper :: Char -> Bool
    IsAsciiUpper,

    // Data.Function operations
    /// on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
    On,
    /// fix :: (a -> a) -> a
    Fix,
    /// (&) :: a -> (a -> b) -> b
    Amp,

    // Data.Maybe additional operations
    /// listToMaybe :: [a] -> Maybe a
    ListToMaybe,
    /// maybeToList :: Maybe a -> [a]
    MaybeToList,
    /// catMaybes :: [Maybe a] -> [a]
    CatMaybes,
    /// mapMaybe :: (a -> Maybe b) -> [a] -> [b]
    MapMaybe,

    // Data.Either additional operations
    /// isLeft :: Either a b -> Bool
    IsLeft,
    /// isRight :: Either a b -> Bool
    IsRight,
    /// lefts :: [Either a b] -> [a]
    Lefts,
    /// rights :: [Either a b] -> [b]
    Rights,
    /// partitionEithers :: [Either a b] -> ([a], [b])
    PartitionEithers,

    // Numeric: math functions
    /// sqrt :: Floating a => a -> a
    Sqrt,
    /// exp :: Floating a => a -> a
    Exp,
    /// log :: Floating a => a -> a
    Log,
    /// sin :: Floating a => a -> a
    Sin,
    /// cos :: Floating a => a -> a
    Cos,
    /// tan :: Floating a => a -> a
    Tan,
    /// (^) :: (Num a, Integral b) => a -> b -> a
    Power,
    /// truncate :: (RealFrac a, Integral b) => a -> b
    Truncate,
    /// round :: (RealFrac a, Integral b) => a -> b
    Round,
    /// ceiling :: (RealFrac a, Integral b) => a -> b
    Ceiling,
    /// floor :: (RealFrac a, Integral b) => a -> b
    Floor,

    // Prelude: fst/snd
    /// fst :: (a, b) -> a
    Fst,
    /// snd :: (a, b) -> b
    Snd,

    // Dictionary operations (generated by type class desugaring)
    /// Select field N from a dictionary (tuple). Generated as `$sel_N` by
    /// HIR-to-Core lowering for type class method extraction.
    DictSelect(usize),
}

impl PrimOp {
    /// Returns the arity of this primitive operation.
    #[must_use]
    pub fn arity(self) -> usize {
        match self {
            // Arity 0
            Self::GetLine | Self::GetChar | Self::GetContents | Self::Otherwise => 0,
            // Arity 1
            Self::NegInt
            | Self::NegDouble
            | Self::NotBool
            | Self::IntToDouble
            | Self::DoubleToInt
            | Self::CharToInt
            | Self::IntToChar
            | Self::Error
            | Self::UArrayFromList
            | Self::UArrayToList
            | Self::UArraySum
            | Self::UArrayLength
            | Self::ListReturn
            | Self::Head
            | Self::Tail
            | Self::Last
            | Self::Init
            | Self::Reverse
            | Self::Null
            | Self::Even
            | Self::Odd
            | Self::Cycle
            | Self::Unzip
            | Self::Product
            | Self::FromIntegral
            | Self::IsJust
            | Self::IsNothing
            | Self::Abs
            | Self::Signum
            | Self::Swap
            | Self::Repeat
            | Self::And
            | Self::Or
            | Self::Lines
            | Self::Unlines
            | Self::Words
            | Self::Unwords
            | Self::Show
            | Self::Id
            | Self::PutStrLn
            | Self::PutStr
            | Self::Print
            | Self::IoReturn
            | Self::DictSelect(_)
            | Self::Succ
            | Self::Pred
            | Self::ToEnum
            | Self::FromEnum
            | Self::Maximum
            | Self::Minimum
            | Self::Unzip3
            | Self::ReadFile
            | Self::RealToFrac
            | Self::Nub
            | Self::Group
            | Self::Transpose
            | Self::Subsequences
            | Self::Permutations
            | Self::Tails
            | Self::Inits
            | Self::GenericLength
            | Self::IsAlpha
            | Self::IsAlphaNum
            | Self::IsAscii
            | Self::IsControl
            | Self::IsDigit
            | Self::IsHexDigit
            | Self::IsLetter
            | Self::IsLower
            | Self::IsNumber
            | Self::IsPrint
            | Self::IsPunctuation
            | Self::IsSpace
            | Self::IsSymbol
            | Self::IsUpper
            | Self::ToLower
            | Self::ToUpper
            | Self::ToTitle
            | Self::DigitToInt
            | Self::IntToDigit
            | Self::IsLatin1
            | Self::IsAsciiLower
            | Self::IsAsciiUpper
            | Self::Fix
            | Self::ListToMaybe
            | Self::MaybeToList
            | Self::CatMaybes
            | Self::IsLeft
            | Self::IsRight
            | Self::Lefts
            | Self::Rights
            | Self::PartitionEithers
            | Self::Sqrt
            | Self::Exp
            | Self::Log
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Truncate
            | Self::Round
            | Self::Ceiling
            | Self::Floor
            | Self::Fst
            | Self::Snd
            | Self::Scanl1
            | Self::Scanr1
            | Self::ShowString
            | Self::ShowChar => 1,
            // Arity 2
            Self::UArrayMap
            | Self::UArrayRange
            | Self::Concat
            | Self::ConcatMap
            | Self::Append
            | Self::ListBind
            | Self::ListThen
            | Self::Filter
            | Self::Zip
            | Self::Take
            | Self::Drop
            | Self::Index
            | Self::Replicate
            | Self::EnumFromTo
            | Self::Elem
            | Self::NotElem
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Span
            | Self::Break
            | Self::SplitAt
            | Self::Iterate
            | Self::Lookup
            | Self::Min
            | Self::Max
            | Self::FromMaybe
            | Self::Any
            | Self::All
            | Self::Const
            | Self::Uncurry
            | Self::IoBind
            | Self::IoThen
            | Self::MonadBind
            | Self::MonadThen
            | Self::Gcd
            | Self::Lcm
            | Self::Quot
            | Self::Rem
            | Self::QuotRem
            | Self::DivMod
            | Self::Subtract
            | Self::WriteFile
            | Self::AppendFile
            | Self::AsTypeOf
            | Self::SortBy
            | Self::SortOn
            | Self::NubBy
            | Self::GroupBy
            | Self::Intersperse
            | Self::Intercalate
            | Self::Partition
            | Self::Find
            | Self::StripPrefix
            | Self::IsPrefixOf
            | Self::IsSuffixOf
            | Self::IsInfixOf
            | Self::Delete
            | Self::Union
            | Self::Intersect
            | Self::ListDiff
            | Self::GenericTake
            | Self::GenericDrop
            | Self::MapMaybe
            | Self::Power
            | Self::On
            | Self::Amp
            | Self::Unfoldr
            | Self::Sort
            | Self::Interact
            | Self::ShowParen => 2,
            // Arity 3
            Self::UArrayZipWith
            | Self::UArrayFold
            | Self::Foldr
            | Self::Foldl
            | Self::FoldlStrict
            | Self::ZipWith
            | Self::Flip
            | Self::MaybeElim
            | Self::EitherElim
            | Self::Curry
            | Self::Scanl
            | Self::Scanr
            | Self::Zip3
            | Self::Until
            | Self::DeleteBy
            | Self::UnionBy
            | Self::IntersectBy
            | Self::MapAccumL
            | Self::MapAccumR => 3,
            // Arity 4
            Self::ZipWith3 => 4,
            // Default arity 2 for arithmetic/comparison ops
            _ => 2,
        }
    }

    /// Looks up a primitive operation by name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "+#" | "plusInt#" => Some(Self::AddInt),
            "-#" | "minusInt#" => Some(Self::SubInt),
            "*#" | "timesInt#" => Some(Self::MulInt),
            "quotInt#" => Some(Self::DivInt),
            "remInt#" => Some(Self::ModInt),
            "negateInt#" => Some(Self::NegInt),
            "+##" | "plusDouble#" => Some(Self::AddDouble),
            "-##" | "minusDouble#" => Some(Self::SubDouble),
            "*##" | "timesDouble#" => Some(Self::MulDouble),
            "/##" | "divideDouble#" => Some(Self::DivDouble),
            "negateDouble#" => Some(Self::NegDouble),
            "==#" | "eqInt#" => Some(Self::EqInt),
            "<#" | "ltInt#" => Some(Self::LtInt),
            "<=#" | "leInt#" => Some(Self::LeInt),
            ">#" | "gtInt#" => Some(Self::GtInt),
            ">=#" | "geInt#" => Some(Self::GeInt),
            "==##" | "eqDouble#" => Some(Self::EqDouble),
            "<##" | "ltDouble#" => Some(Self::LtDouble),
            "andBool" => Some(Self::AndBool),
            "orBool" => Some(Self::OrBool),
            "not" => Some(Self::NotBool),
            "int2Double#" => Some(Self::IntToDouble),
            "double2Int#" => Some(Self::DoubleToInt),
            "eqChar#" => Some(Self::EqChar),
            "ord" | "ord#" => Some(Self::CharToInt),
            "chr" | "chr#" => Some(Self::IntToChar),
            "seq" => Some(Self::Seq),
            "error" => Some(Self::Error),
            // UArray operations
            "uarrayFromList" | "fromList" => Some(Self::UArrayFromList),
            "uarrayToList" | "toList" => Some(Self::UArrayToList),
            "uarrayMap" | "map" => Some(Self::UArrayMap),
            "uarrayZipWith" => Some(Self::UArrayZipWith),
            "uarrayFold" => Some(Self::UArrayFold),
            "uarraySum" | "sum" => Some(Self::UArraySum),
            "uarrayLength" | "length" => Some(Self::UArrayLength),
            "uarrayRange" | "range" => Some(Self::UArrayRange),
            // List operations
            "++" | "concat" => Some(Self::Concat),
            "concatMap" => Some(Self::ConcatMap),
            "append" => Some(Self::Append),
            // Monad operations (list monad for now)
            ">>=" => Some(Self::ListBind),
            ">>" => Some(Self::ListThen),
            "return" => Some(Self::ListReturn),
            // Additional list operations
            "foldr" => Some(Self::Foldr),
            "foldl" => Some(Self::Foldl),
            "foldl'" => Some(Self::FoldlStrict),
            "filter" => Some(Self::Filter),
            "zip" => Some(Self::Zip),
            "zipWith" => Some(Self::ZipWith),
            "take" => Some(Self::Take),
            "drop" => Some(Self::Drop),
            "head" => Some(Self::Head),
            "tail" => Some(Self::Tail),
            "last" => Some(Self::Last),
            "init" => Some(Self::Init),
            "reverse" => Some(Self::Reverse),
            "null" => Some(Self::Null),
            "!!" => Some(Self::Index),
            "replicate" => Some(Self::Replicate),
            "enumFromTo" => Some(Self::EnumFromTo),
            // Additional list/prelude operations
            "even" => Some(Self::Even),
            "odd" => Some(Self::Odd),
            "elem" => Some(Self::Elem),
            "notElem" => Some(Self::NotElem),
            "takeWhile" => Some(Self::TakeWhile),
            "dropWhile" => Some(Self::DropWhile),
            "span" => Some(Self::Span),
            "break" => Some(Self::Break),
            "splitAt" => Some(Self::SplitAt),
            "iterate" => Some(Self::Iterate),
            "repeat" => Some(Self::Repeat),
            "cycle" => Some(Self::Cycle),
            "lookup" => Some(Self::Lookup),
            "unzip" => Some(Self::Unzip),
            "product" => Some(Self::Product),
            "flip" => Some(Self::Flip),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "fromIntegral" | "toInteger" => Some(Self::FromIntegral),
            "maybe" => Some(Self::MaybeElim),
            "fromMaybe" => Some(Self::FromMaybe),
            "either" => Some(Self::EitherElim),
            "isJust" => Some(Self::IsJust),
            "isNothing" => Some(Self::IsNothing),
            "abs" => Some(Self::Abs),
            "signum" => Some(Self::Signum),
            "curry" => Some(Self::Curry),
            "uncurry" => Some(Self::Uncurry),
            "swap" => Some(Self::Swap),
            "any" => Some(Self::Any),
            "all" => Some(Self::All),
            "and" => Some(Self::And),
            "or" => Some(Self::Or),
            "lines" => Some(Self::Lines),
            "unlines" => Some(Self::Unlines),
            "words" => Some(Self::Words),
            "unwords" => Some(Self::Unwords),
            "show" => Some(Self::Show),
            "id" => Some(Self::Id),
            "const" => Some(Self::Const),
            // IO operations
            "putStrLn" => Some(Self::PutStrLn),
            "putStr" => Some(Self::PutStr),
            "print" => Some(Self::Print),
            "getLine" => Some(Self::GetLine),
            // Enum operations
            "succ" => Some(Self::Succ),
            "pred" => Some(Self::Pred),
            "toEnum" => Some(Self::ToEnum),
            "fromEnum" => Some(Self::FromEnum),
            // Integral operations
            "gcd" => Some(Self::Gcd),
            "lcm" => Some(Self::Lcm),
            "quot" => Some(Self::Quot),
            "rem" => Some(Self::Rem),
            "quotRem" => Some(Self::QuotRem),
            "divMod" => Some(Self::DivMod),
            "subtract" => Some(Self::Subtract),
            // Scan operations
            "scanl" => Some(Self::Scanl),
            "scanl'" => Some(Self::Scanl),
            "scanr" => Some(Self::Scanr),
            "scanl1" => Some(Self::Scanl1),
            "scanr1" => Some(Self::Scanr1),
            // More list operations
            "maximum" => Some(Self::Maximum),
            "minimum" => Some(Self::Minimum),
            "zip3" => Some(Self::Zip3),
            "zipWith3" => Some(Self::ZipWith3),
            "unzip3" => Some(Self::Unzip3),
            // Show helpers
            "showString" => Some(Self::ShowString),
            "showChar" => Some(Self::ShowChar),
            "showParen" => Some(Self::ShowParen),
            // IO operations (additional)
            "getChar" => Some(Self::GetChar),
            "getContents" => Some(Self::GetContents),
            "readFile" => Some(Self::ReadFile),
            "writeFile" => Some(Self::WriteFile),
            "appendFile" => Some(Self::AppendFile),
            "interact" => Some(Self::Interact),
            // Misc Prelude
            "otherwise" => Some(Self::Otherwise),
            "until" => Some(Self::Until),
            "asTypeOf" => Some(Self::AsTypeOf),
            "realToFrac" => Some(Self::RealToFrac),
            // Data.List
            "sort" => Some(Self::Sort),
            "sortBy" => Some(Self::SortBy),
            "sortOn" => Some(Self::SortOn),
            "nub" => Some(Self::Nub),
            "nubBy" => Some(Self::NubBy),
            "group" => Some(Self::Group),
            "groupBy" => Some(Self::GroupBy),
            "intersperse" => Some(Self::Intersperse),
            "intercalate" => Some(Self::Intercalate),
            "transpose" => Some(Self::Transpose),
            "subsequences" => Some(Self::Subsequences),
            "permutations" => Some(Self::Permutations),
            "partition" => Some(Self::Partition),
            "find" => Some(Self::Find),
            "stripPrefix" => Some(Self::StripPrefix),
            "isPrefixOf" => Some(Self::IsPrefixOf),
            "isSuffixOf" => Some(Self::IsSuffixOf),
            "isInfixOf" => Some(Self::IsInfixOf),
            "delete" => Some(Self::Delete),
            "deleteBy" => Some(Self::DeleteBy),
            "union" => Some(Self::Union),
            "unionBy" => Some(Self::UnionBy),
            "intersect" => Some(Self::Intersect),
            "intersectBy" => Some(Self::IntersectBy),
            "\\\\" => Some(Self::ListDiff),
            "tails" => Some(Self::Tails),
            "inits" => Some(Self::Inits),
            "mapAccumL" => Some(Self::MapAccumL),
            "mapAccumR" => Some(Self::MapAccumR),
            "unfoldr" => Some(Self::Unfoldr),
            "genericLength" => Some(Self::GenericLength),
            "genericTake" => Some(Self::GenericTake),
            "genericDrop" => Some(Self::GenericDrop),
            // Data.Char
            "isAlpha" => Some(Self::IsAlpha),
            "isAlphaNum" => Some(Self::IsAlphaNum),
            "isAscii" => Some(Self::IsAscii),
            "isControl" => Some(Self::IsControl),
            "isDigit" => Some(Self::IsDigit),
            "isHexDigit" => Some(Self::IsHexDigit),
            "isLetter" => Some(Self::IsLetter),
            "isLower" => Some(Self::IsLower),
            "isNumber" => Some(Self::IsNumber),
            "isPrint" => Some(Self::IsPrint),
            "isPunctuation" => Some(Self::IsPunctuation),
            "isSpace" => Some(Self::IsSpace),
            "isSymbol" => Some(Self::IsSymbol),
            "isUpper" => Some(Self::IsUpper),
            "toLower" => Some(Self::ToLower),
            "toUpper" => Some(Self::ToUpper),
            "toTitle" => Some(Self::ToTitle),
            "digitToInt" => Some(Self::DigitToInt),
            "intToDigit" => Some(Self::IntToDigit),
            "isLatin1" => Some(Self::IsLatin1),
            "isAsciiLower" => Some(Self::IsAsciiLower),
            "isAsciiUpper" => Some(Self::IsAsciiUpper),
            // Data.Function
            "on" => Some(Self::On),
            "fix" => Some(Self::Fix),
            "&" => Some(Self::Amp),
            // Data.Maybe additional
            "listToMaybe" => Some(Self::ListToMaybe),
            "maybeToList" => Some(Self::MaybeToList),
            "catMaybes" => Some(Self::CatMaybes),
            "mapMaybe" => Some(Self::MapMaybe),
            // Data.Either additional
            "isLeft" => Some(Self::IsLeft),
            "isRight" => Some(Self::IsRight),
            "lefts" => Some(Self::Lefts),
            "rights" => Some(Self::Rights),
            "partitionEithers" => Some(Self::PartitionEithers),
            // Math functions
            "sqrt" => Some(Self::Sqrt),
            "exp" => Some(Self::Exp),
            "log" => Some(Self::Log),
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "tan" => Some(Self::Tan),
            "^" => Some(Self::Power),
            "truncate" => Some(Self::Truncate),
            "round" => Some(Self::Round),
            "ceiling" => Some(Self::Ceiling),
            "floor" => Some(Self::Floor),
            // Tuple
            "fst" => Some(Self::Fst),
            "snd" => Some(Self::Snd),
            _ => None,
        }
    }
}
