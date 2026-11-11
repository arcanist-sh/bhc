//! `read`, `readMaybe` and `reads` at a type the compiler describes.
//!
//! The compiler cannot yet compile a `Read` class from source, so `read` at a
//! data type, a tuple, a nested list or a `Maybe` had no parser at all. Here
//! the compiler passes a description of the target type and the runtime parses
//! the text against it, building values in the layouts compiled code uses:
//!
//! * a constructor is `[tag: i64, field0, field1, …]`, a nullary one its tag
//!   alone; tuples are a constructor with tag 0, `()` is tag 0 with no field;
//! * a list is a cons cell `[1, head, tail]` or a nil cell `[0]`;
//! * `Int` and `Char` are the machine word itself, `Double` its bits;
//! * `Integer` is a boxed bignum; a newtype is its field.
//!
//! # The description
//!
//! Whitespace-separated s-expressions: the target type first, then the data
//! types it refers to, numbered from 0 in order.
//!
//! ```text
//! type ::= i | n | d | c | s | b | o | u        Int Integer Double Char String Bool Ordering ()
//!        | (l type) | (t type type …)           list, tuple
//!        | (m type) | (e type type)             Maybe, Either
//!        | @k                                   data type number k
//! data ::= (D con …) | (N con)                  data, newtype
//! con  ::= (C Name tag P type …)                positional fields
//!        | (C Name tag R field type …)          record fields
//! ```
//!
//! # Syntax accepted
//!
//! What a derived `Read` accepts: constructor application needs parentheses
//! where precedence 10 would (`Just (Just 1)`), a negative number where
//! precedence 6 would (`Just (-1)`), record syntax binds tightest, and any
//! value may be wrapped in extra parentheses.

use std::ffi::{c_char, CStr, CString};

use crate::ffi::{alloc_cons, alloc_nil, bhc_force, bhc_integer_from_str};

/// A type to read at.
#[derive(Clone, Debug, PartialEq)]
enum Shape {
    Int,
    Integer,
    Double,
    Char,
    Str,
    Bool,
    Ordering,
    Unit,
    List(Box<Shape>),
    Tuple(Vec<Shape>),
    Maybe(Box<Shape>),
    Either(Box<Shape>, Box<Shape>),
    Data(usize),
}

/// One constructor of a described data type.
#[derive(Clone, Debug, PartialEq)]
struct Con {
    name: String,
    tag: i64,
    /// Field names when the constructor was declared with record syntax.
    record: Option<Vec<String>>,
    fields: Vec<Shape>,
}

/// A described data type.
#[derive(Clone, Debug, PartialEq)]
struct Data {
    newtype: bool,
    cons: Vec<Con>,
}

/// A parsed description: the target and the data types it refers to.
#[derive(Clone, Debug, PartialEq)]
struct Description {
    root: Shape,
    datas: Vec<Data>,
}

// ----------------------------------------------------------------------------
// Parsing the description
// ----------------------------------------------------------------------------

#[derive(Debug)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

fn parse_sexps(text: &str) -> Option<Vec<Sexp>> {
    let mut stack: Vec<Vec<Sexp>> = vec![Vec::new()];
    let mut atom = String::new();
    let flush = |atom: &mut String, stack: &mut Vec<Vec<Sexp>>| {
        if !atom.is_empty() {
            if let Some(top) = stack.last_mut() {
                top.push(Sexp::Atom(std::mem::take(atom)));
            }
        }
    };
    for ch in text.chars() {
        match ch {
            '(' => {
                flush(&mut atom, &mut stack);
                stack.push(Vec::new());
            }
            ')' => {
                flush(&mut atom, &mut stack);
                let done = stack.pop()?;
                stack.last_mut()?.push(Sexp::List(done));
            }
            c if c.is_whitespace() => flush(&mut atom, &mut stack),
            c => atom.push(c),
        }
    }
    flush(&mut atom, &mut stack);
    if stack.len() == 1 {
        stack.pop()
    } else {
        None
    }
}

fn shape_of(sexp: &Sexp) -> Option<Shape> {
    match sexp {
        Sexp::Atom(a) => Some(match a.as_str() {
            "i" => Shape::Int,
            "n" => Shape::Integer,
            "d" => Shape::Double,
            "c" => Shape::Char,
            "s" => Shape::Str,
            "b" => Shape::Bool,
            "o" => Shape::Ordering,
            "u" => Shape::Unit,
            other => Shape::Data(other.strip_prefix('@')?.parse().ok()?),
        }),
        Sexp::List(items) => {
            let (Sexp::Atom(head), rest) = items.split_first()? else {
                return None;
            };
            let shapes: Option<Vec<Shape>> = rest.iter().map(shape_of).collect();
            let mut shapes = shapes?;
            match (head.as_str(), shapes.len()) {
                ("l", 1) => Some(Shape::List(Box::new(shapes.remove(0)))),
                ("m", 1) => Some(Shape::Maybe(Box::new(shapes.remove(0)))),
                ("e", 2) => {
                    let right = shapes.remove(1);
                    Some(Shape::Either(Box::new(shapes.remove(0)), Box::new(right)))
                }
                ("t", n) if n >= 2 => Some(Shape::Tuple(shapes)),
                _ => None,
            }
        }
    }
}

fn con_of(sexp: &Sexp) -> Option<Con> {
    let Sexp::List(items) = sexp else {
        return None;
    };
    let [Sexp::Atom(c), Sexp::Atom(name), Sexp::Atom(tag), Sexp::Atom(kind), rest @ ..] =
        items.as_slice()
    else {
        return None;
    };
    if c != "C" {
        return None;
    }
    let tag = tag.parse().ok()?;
    match kind.as_str() {
        "P" => Some(Con {
            name: name.clone(),
            tag,
            record: None,
            fields: rest.iter().map(shape_of).collect::<Option<_>>()?,
        }),
        "R" => {
            let mut names = Vec::new();
            let mut fields = Vec::new();
            for pair in rest.chunks(2) {
                let [Sexp::Atom(field), ty] = pair else {
                    return None;
                };
                names.push(field.clone());
                fields.push(shape_of(ty)?);
            }
            Some(Con {
                name: name.clone(),
                tag,
                record: Some(names),
                fields,
            })
        }
        _ => None,
    }
}

fn data_of(sexp: &Sexp) -> Option<Data> {
    let Sexp::List(items) = sexp else {
        return None;
    };
    let (Sexp::Atom(kind), cons) = items.split_first()? else {
        return None;
    };
    let cons: Vec<Con> = cons.iter().map(con_of).collect::<Option<_>>()?;
    match kind.as_str() {
        "D" => Some(Data {
            newtype: false,
            cons,
        }),
        "N" if cons.len() == 1 && cons[0].fields.len() == 1 => Some(Data {
            newtype: true,
            cons,
        }),
        _ => None,
    }
}

fn parse_description(text: &str) -> Option<Description> {
    let sexps = parse_sexps(text)?;
    let (root, datas) = sexps.split_first()?;
    let description = Description {
        root: shape_of(root)?,
        datas: datas.iter().map(data_of).collect::<Option<_>>()?,
    };
    // Every reference must name a described data type.
    fn refs_ok(shape: &Shape, count: usize) -> bool {
        match shape {
            Shape::Data(k) => *k < count,
            Shape::List(e) | Shape::Maybe(e) => refs_ok(e, count),
            Shape::Either(a, b) => refs_ok(a, count) && refs_ok(b, count),
            Shape::Tuple(es) => es.iter().all(|e| refs_ok(e, count)),
            _ => true,
        }
    }
    let count = description.datas.len();
    let all_ok = refs_ok(&description.root, count)
        && description
            .datas
            .iter()
            .flat_map(|d| &d.cons)
            .flat_map(|c| &c.fields)
            .all(|f| refs_ok(f, count));
    all_ok.then_some(description)
}

// ----------------------------------------------------------------------------
// Parsing the text
// ----------------------------------------------------------------------------

/// A parsed value, independent of the heap, so the parser can be tested
/// without building runtime objects.
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Int(i64),
    Integer(String),
    Double(f64),
    Char(char),
    Str(String),
    List(Vec<Value>),
    /// A constructor (also tuples, `()`, `Bool`, `Maybe`, …): tag and fields.
    Con(i64, Vec<Value>),
}

struct Reader<'a> {
    description: &'a Description,
    chars: &'a [char],
    pos: usize,
}

/// The names of the ASCII control characters, as `\NUL` … `\US`, by code.
const ASCII_NAMES: [&str; 33] = [
    "NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL", "BS", "HT", "LF", "VT", "FF", "CR",
    "SO", "SI", "DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB", "CAN", "EM", "SUB", "ESC",
    "FS", "GS", "RS", "US", "SP",
];

impl Reader<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.pos += 1;
        }
    }

    /// Consume `ch` after optional whitespace.
    fn symbol(&mut self, ch: char) -> Option<()> {
        self.skip_space();
        (self.peek()? == ch).then(|| self.pos += 1)
    }

    /// Run `f`, restoring the position if it fails.
    fn attempt<T>(&mut self, f: impl FnOnce(&mut Self) -> Option<T>) -> Option<T> {
        let saved = self.pos;
        let result = f(self);
        if result.is_none() {
            self.pos = saved;
        }
        result
    }

    /// An identifier token, whole: `Greenish` is not `Green`.
    fn identifier(&mut self) -> Option<String> {
        self.skip_space();
        let start = self.pos;
        if !self.peek().is_some_and(|c| c.is_alphabetic() || c == '_') {
            return None;
        }
        while self
            .peek()
            .is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '\'')
        {
            self.pos += 1;
        }
        Some(self.chars[start..self.pos].iter().collect())
    }

    fn keyword(&mut self, word: &str) -> Option<()> {
        self.attempt(|r| (r.identifier()? == word).then_some(()))
    }

    fn digits(&mut self, radix: u32) -> Option<String> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_digit(radix)) {
            self.pos += 1;
        }
        (self.pos > start).then(|| self.chars[start..self.pos].iter().collect())
    }

    /// The sign of a number: negative literals need precedence at most 6.
    fn sign(&mut self, prec: u8) -> Option<bool> {
        self.skip_space();
        if self.peek() == Some('-') {
            if prec > 6 {
                return None;
            }
            self.pos += 1;
            self.skip_space();
            Some(true)
        } else {
            Some(false)
        }
    }

    /// An integer literal: decimal, `0x…` or `0o…`. Radix and digits.
    fn integer_literal(&mut self) -> Option<(u32, String)> {
        if self.peek() == Some('0') {
            let radix = match self.chars.get(self.pos + 1) {
                Some('x' | 'X') => Some(16),
                Some('o' | 'O') => Some(8),
                _ => None,
            };
            if let Some(radix) = radix {
                let found = self.attempt(|r| {
                    r.pos += 2;
                    r.digits(radix)
                });
                if let Some(digits) = found {
                    return Some((radix, digits));
                }
            }
        }
        Some((10, self.digits(10)?))
    }

    /// A number token must not run into an identifier (`12abc`).
    fn end_of_number(&self) -> Option<()> {
        (!self.peek().is_some_and(|c| c.is_alphanumeric() || c == '_')).then_some(())
    }

    fn int(&mut self, prec: u8) -> Option<Value> {
        let negative = self.sign(prec)?;
        let (radix, digits) = self.integer_literal()?;
        self.end_of_number()?;
        // `read` at `Int` wraps like `fromInteger`.
        let mut n: i64 = 0;
        for d in digits.chars() {
            n = n
                .wrapping_mul(i64::from(radix))
                .wrapping_add(i64::from(d.to_digit(radix)?));
        }
        Some(Value::Int(if negative { n.wrapping_neg() } else { n }))
    }

    fn integer(&mut self, prec: u8) -> Option<Value> {
        let negative = self.sign(prec)?;
        let (radix, digits) = self.integer_literal()?;
        self.end_of_number()?;
        let decimal = if radix == 10 {
            digits
        } else {
            num_bigint::BigInt::parse_bytes(digits.as_bytes(), radix)?.to_string()
        };
        Some(Value::Integer(if negative {
            format!("-{decimal}")
        } else {
            decimal
        }))
    }

    fn double(&mut self, prec: u8) -> Option<Value> {
        let negative = self.sign(prec)?;
        let magnitude = if self.keyword("NaN").is_some() {
            f64::NAN
        } else if self.keyword("Infinity").is_some() {
            f64::INFINITY
        } else {
            let mut text = self.digits(10)?;
            let fraction = self.attempt(|r| {
                (r.peek()? == '.').then(|| r.pos += 1)?;
                r.digits(10)
            });
            if let Some(fraction) = fraction {
                text.push('.');
                text.push_str(&fraction);
            }
            let exponent = self.attempt(|r| {
                matches!(r.peek()?, 'e' | 'E').then(|| r.pos += 1)?;
                let sign = match r.peek()? {
                    '-' => {
                        r.pos += 1;
                        "-"
                    }
                    '+' => {
                        r.pos += 1;
                        ""
                    }
                    _ => "",
                };
                Some(format!("e{sign}{}", r.digits(10)?))
            });
            if let Some(exponent) = exponent {
                text.push_str(&exponent);
            }
            self.end_of_number()?;
            text.parse::<f64>().ok()?
        };
        Some(Value::Double(if negative { -magnitude } else { magnitude }))
    }

    /// One character of a character or string literal, after any escape.
    /// `None` inside `Some` is the empty escape `\&` or a string gap.
    fn literal_char(&mut self, quote: char) -> Option<Option<char>> {
        let ch = self.peek()?;
        if ch == quote || ch == '\n' {
            return None;
        }
        self.pos += 1;
        if ch != '\\' {
            return Some(Some(ch));
        }
        let escape = self.peek()?;
        let simple = match escape {
            'n' => Some('\n'),
            't' => Some('\t'),
            'r' => Some('\r'),
            'a' => Some('\x07'),
            'b' => Some('\x08'),
            'f' => Some('\x0c'),
            'v' => Some('\x0b'),
            '\\' | '"' | '\'' => Some(escape),
            _ => None,
        };
        if let Some(c) = simple {
            self.pos += 1;
            return Some(Some(c));
        }
        if escape == '&' && quote == '"' {
            self.pos += 1;
            return Some(None);
        }
        if escape.is_whitespace() && quote == '"' {
            // A string gap: backslash, whitespace, backslash.
            self.skip_space();
            self.symbol('\\')?;
            return Some(None);
        }
        if escape == '^' {
            let control = *self.chars.get(self.pos + 1)?;
            if ('@'..='_').contains(&control) {
                self.pos += 2;
                return Some(char::from_u32(control as u32 - 64));
            }
            return None;
        }
        let code = match escape {
            'x' | 'X' => {
                self.pos += 1;
                u32::from_str_radix(&self.digits(16)?, 16).ok()?
            }
            'o' | 'O' => {
                self.pos += 1;
                u32::from_str_radix(&self.digits(8)?, 8).ok()?
            }
            c if c.is_ascii_digit() => self.digits(10)?.parse().ok()?,
            _ => {
                // A named control character; the longest name wins (`SOH`
                // before `SO`).
                let rest: String = self.chars[self.pos..].iter().take(3).collect();
                let (code, name) = ASCII_NAMES
                    .iter()
                    .enumerate()
                    .filter(|(_, name)| rest.starts_with(**name))
                    .max_by_key(|(_, name)| name.len())
                    .map(|(code, name)| (code as u32, *name))
                    .or_else(|| rest.starts_with("DEL").then_some((127, "DEL")))?;
                self.pos += name.len();
                code
            }
        };
        Some(Some(char::from_u32(code)?))
    }

    fn character(&mut self) -> Option<Value> {
        self.symbol('\'')?;
        let ch = self.literal_char('\'')??;
        (self.peek()? == '\'').then(|| self.pos += 1)?;
        Some(Value::Char(ch))
    }

    fn string(&mut self) -> Option<Value> {
        self.skip_space();
        if self.peek() == Some('[') {
            // A string is a list of characters and may be written as one.
            let Value::List(items) = self.list(&Shape::Char)? else {
                return None;
            };
            let text: Option<String> = items
                .into_iter()
                .map(|v| match v {
                    Value::Char(c) => Some(c),
                    _ => None,
                })
                .collect();
            return Some(Value::Str(text?));
        }
        self.symbol('"')?;
        let mut text = String::new();
        while self.peek()? != '"' {
            if let Some(ch) = self.literal_char('"')? {
                text.push(ch);
            }
        }
        self.pos += 1;
        Some(Value::Str(text))
    }

    fn list(&mut self, elem: &Shape) -> Option<Value> {
        self.symbol('[')?;
        let mut items = Vec::new();
        if self.symbol(']').is_some() {
            return Some(Value::List(items));
        }
        loop {
            items.push(self.value(elem, 0)?);
            if self.symbol(',').is_none() {
                break;
            }
        }
        self.symbol(']')?;
        Some(Value::List(items))
    }

    fn tuple(&mut self, elems: &[Shape]) -> Option<Value> {
        self.symbol('(')?;
        let mut fields = Vec::with_capacity(elems.len());
        for (i, elem) in elems.iter().enumerate() {
            if i > 0 {
                self.symbol(',')?;
            }
            fields.push(self.value(elem, 0)?);
        }
        self.symbol(')')?;
        Some(Value::Con(0, fields))
    }

    /// One of `names`, as a nullary constructor whose tag is its position.
    fn enumeration(&mut self, names: &[&str]) -> Option<Value> {
        let word = self.identifier()?;
        let tag = names.iter().position(|n| *n == word)?;
        Some(Value::Con(tag as i64, Vec::new()))
    }

    /// A constructor applied to positional arguments.
    fn application(&mut self, name: &str, tag: i64, fields: &[Shape], prec: u8) -> Option<Value> {
        self.keyword(name)?;
        if !fields.is_empty() && prec > 10 {
            return None;
        }
        let args: Option<Vec<Value>> = fields.iter().map(|f| self.value(f, 11)).collect();
        Some(Value::Con(tag, args?))
    }

    fn record(&mut self, con: &Con, names: &[String], prec: u8) -> Option<Value> {
        self.keyword(&con.name)?;
        if prec > 11 {
            return None;
        }
        self.symbol('{')?;
        let mut fields = Vec::with_capacity(names.len());
        for (i, (name, shape)) in names.iter().zip(&con.fields).enumerate() {
            if i > 0 {
                self.symbol(',')?;
            }
            self.keyword(name)?;
            self.symbol('=')?;
            fields.push(self.value(shape, 0)?);
        }
        self.symbol('}')?;
        Some(Value::Con(con.tag, fields))
    }

    fn data(&mut self, index: usize, prec: u8) -> Option<Value> {
        let description = self.description;
        let data = description.datas.get(index)?;
        for con in &data.cons {
            let parsed = self.attempt(|r| match &con.record {
                Some(names) => r.record(con, names, prec),
                None => r.application(&con.name, con.tag, &con.fields, prec),
            });
            if let Some(value) = parsed {
                return Some(match value {
                    // A newtype is its field.
                    Value::Con(_, mut fields) if data.newtype => fields.pop()?,
                    other => other,
                });
            }
        }
        None
    }

    /// The value as written, without surrounding parentheses.
    fn bare(&mut self, shape: &Shape, prec: u8) -> Option<Value> {
        match shape {
            Shape::Int => self.int(prec),
            Shape::Integer => self.integer(prec),
            Shape::Double => self.double(prec),
            Shape::Char => self.character(),
            Shape::Str => self.string(),
            Shape::Bool => self.enumeration(&["False", "True"]),
            Shape::Ordering => self.enumeration(&["LT", "EQ", "GT"]),
            Shape::Unit => {
                self.symbol('(')?;
                self.symbol(')')?;
                Some(Value::Con(0, Vec::new()))
            }
            Shape::List(elem) => self.list(elem),
            Shape::Tuple(elems) => self.tuple(elems),
            Shape::Maybe(elem) => self
                .attempt(|r| r.application("Nothing", 0, &[], prec))
                .or_else(|| {
                    self.attempt(|r| r.application("Just", 1, std::slice::from_ref(elem), prec))
                }),
            Shape::Either(left, right) => self
                .attempt(|r| r.application("Left", 0, std::slice::from_ref(left), prec))
                .or_else(|| {
                    self.attempt(|r| r.application("Right", 1, std::slice::from_ref(right), prec))
                }),
            Shape::Data(index) => self.data(*index, prec),
        }
    }

    /// A value at `prec`, with any number of surrounding parentheses.
    fn value(&mut self, shape: &Shape, prec: u8) -> Option<Value> {
        if let Some(value) = self.attempt(|r| r.bare(shape, prec)) {
            return Some(value);
        }
        self.attempt(|r| {
            r.symbol('(')?;
            let value = r.value(shape, 0)?;
            r.symbol(')')?;
            Some(value)
        })
    }
}

/// Parse a value of the described type from the start of `chars`: the value
/// and how many characters it took.
fn read_prefix(description: &Description, chars: &[char]) -> Option<(Value, usize)> {
    let mut reader = Reader {
        description,
        chars,
        pos: 0,
    };
    let value = reader.value(&description.root, 0)?;
    Some((value, reader.pos))
}

/// Parse a value that is the whole of `chars`, but for surrounding whitespace.
fn read_all(description: &Description, chars: &[char]) -> Option<Value> {
    let (value, used) = read_prefix(description, chars)?;
    chars[used..]
        .iter()
        .all(|c| c.is_whitespace())
        .then_some(value)
}

// ----------------------------------------------------------------------------
// Building runtime values
// ----------------------------------------------------------------------------

/// Allocate `[tag, fields…]`.
unsafe fn alloc_con(tag: i64, fields: &[*mut u8]) -> *mut u8 {
    unsafe {
        let size = 8 + 8 * fields.len();
        let layout = std::alloc::Layout::from_size_align_unchecked(size, 8);
        let obj = std::alloc::alloc(layout);
        *(obj as *mut i64) = tag;
        for (i, field) in fields.iter().enumerate() {
            *(obj.add(8 + 8 * i) as *mut *mut u8) = *field;
        }
        obj
    }
}

unsafe fn build_string(text: &str) -> *mut u8 {
    unsafe {
        let mut list = alloc_nil();
        for ch in text.chars().rev() {
            list = alloc_cons(ch as usize as *mut u8, list);
        }
        list
    }
}

unsafe fn build(value: &Value) -> *mut u8 {
    unsafe {
        match value {
            Value::Int(n) => *n as usize as *mut u8,
            Value::Double(d) => d.to_bits() as usize as *mut u8,
            Value::Char(c) => *c as usize as *mut u8,
            Value::Integer(decimal) => match CString::new(decimal.as_str()) {
                Ok(text) => bhc_integer_from_str(text.as_ptr()),
                Err(_) => std::ptr::null_mut(),
            },
            Value::Str(text) => build_string(text),
            Value::List(items) => {
                let mut list = alloc_nil();
                for item in items.iter().rev() {
                    list = alloc_cons(build(item), list);
                }
                list
            }
            Value::Con(tag, fields) => {
                let fields: Vec<*mut u8> = fields.iter().map(|f| build(f)).collect();
                alloc_con(*tag, &fields)
            }
        }
    }
}

/// The cons cells of a string, and its characters.
unsafe fn string_cells(str_list: *mut u8) -> (Vec<*mut u8>, Vec<char>) {
    unsafe {
        let tag = |p: *mut u8| *(p as *const i64);
        let field = |p: *mut u8, i: usize| *(p.add(8 + i * 8) as *const *mut u8);
        let mut cells = Vec::new();
        let mut chars = Vec::new();
        let mut list = bhc_force(str_list);
        while tag(list) != 0 {
            cells.push(list);
            // A character is the code point itself, not a heap object.
            let code = field(list, 0) as usize;
            chars.push(
                u32::try_from(code)
                    .ok()
                    .and_then(char::from_u32)
                    .unwrap_or('\u{fffd}'),
            );
            list = bhc_force(field(list, 1));
        }
        (cells, chars)
    }
}

/// `read` (`mode` 0), `readMaybe` (1) or `reads` (2) of `str_list` at the
/// type `description` describes.
///
/// `read` exits with `Prelude.read: no parse` when the text is not a value of
/// the type; `readMaybe` answers `Nothing`; `reads` answers `[]`, or
/// `[(value, rest)]` with the unread remainder of the string.
///
/// # Safety
///
/// `description` must be a NUL-terminated string in the format this module
/// documents, and `str_list` a BHC list of characters or a thunk evaluating
/// to one.
#[no_mangle]
pub unsafe extern "C" fn bhc_read_generic(
    description: *const c_char,
    str_list: *mut u8,
    mode: i64,
) -> *mut u8 {
    unsafe {
        let text = CStr::from_ptr(description).to_str().unwrap_or("");
        let Some(description) = parse_description(text) else {
            crate::ffi::rts_abort(&format!("read: malformed type description `{text}`"));
        };
        let (cells, chars) = string_cells(str_list);
        match mode {
            1 => match read_all(&description, &chars) {
                Some(value) => alloc_con(1, &[build(&value)]),
                None => alloc_con(0, &[]),
            },
            2 => match read_prefix(&description, &chars) {
                Some((value, used)) => {
                    let rest = cells.get(used).copied().unwrap_or_else(|| alloc_nil());
                    let pair = alloc_con(0, &[build(&value), rest]);
                    alloc_cons(pair, alloc_nil())
                }
                None => alloc_nil(),
            },
            _ => match read_all(&description, &chars) {
                Some(value) => build(&value),
                None => {
                    eprintln!("Prelude.read: no parse");
                    std::process::exit(1);
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(description: &str, text: &str) -> Option<Value> {
        let description = parse_description(description).expect("description parses");
        let chars: Vec<char> = text.chars().collect();
        read_all(&description, &chars)
    }

    fn con(tag: i64, fields: Vec<Value>) -> Value {
        Value::Con(tag, fields)
    }

    const COLOR: &str = "@0 (D (C Red 0 P) (C Green 1 P) (C Blue 2 P))";
    const PERSON: &str = "@0 (D (C P 0 R name s age i))";
    const TREE: &str = "@0 (D (C Leaf 0 P) (C Node 1 P @0 i @0))";

    #[test]
    fn description_rejects_malformed_input() {
        assert!(parse_description("").is_none());
        assert!(parse_description("(l i").is_none());
        assert!(parse_description("@1 (D (C A 0 P))").is_none());
        assert!(parse_description("(t i)").is_none());
        assert!(parse_description("@0 (N (C A 0 P i i))").is_none());
    }

    #[test]
    fn reads_numbers() {
        assert_eq!(read("i", " 42 "), Some(Value::Int(42)));
        assert_eq!(read("i", "-7"), Some(Value::Int(-7)));
        assert_eq!(read("i", "(-7)"), Some(Value::Int(-7)));
        assert_eq!(read("i", "0x1F"), Some(Value::Int(31)));
        assert_eq!(read("i", "0o17"), Some(Value::Int(15)));
        assert_eq!(read("i", "12abc"), None);
        assert_eq!(read("i", "1.5"), None);
        assert_eq!(
            read("n", "-123456789012345678901234567890"),
            Some(Value::Integer("-123456789012345678901234567890".into()))
        );
        assert_eq!(read("d", "3"), Some(Value::Double(3.0)));
        assert_eq!(read("d", "-2.5e-3"), Some(Value::Double(-2.5e-3)));
        assert_eq!(read("d", "Infinity"), Some(Value::Double(f64::INFINITY)));
        assert_eq!(read("d", "."), None);
    }

    #[test]
    fn reads_characters_and_strings() {
        assert_eq!(read("c", "'a'"), Some(Value::Char('a')));
        assert_eq!(read("c", "'\\n'"), Some(Value::Char('\n')));
        assert_eq!(read("c", "'\\''"), Some(Value::Char('\'')));
        assert_eq!(read("c", "'ab'"), None);
        assert_eq!(
            read("s", "\"a\\tb\\\"c\\1234\\x41\\SOH\\&1\""),
            Some(Value::Str("a\tb\"c\u{4d2}A\u{1}1".into()))
        );
        assert_eq!(read("s", "['h','i']"), Some(Value::Str("hi".into())));
        assert_eq!(read("s", "\"open"), None);
    }

    #[test]
    fn reads_containers() {
        assert_eq!(
            read("(l (t i s))", "[(1,\"a\"),(2,\"b\")]"),
            Some(Value::List(vec![
                con(0, vec![Value::Int(1), Value::Str("a".into())]),
                con(0, vec![Value::Int(2), Value::Str("b".into())]),
            ]))
        );
        assert_eq!(read("(l i)", "[ ]"), Some(Value::List(vec![])));
        assert_eq!(read("(l i)", "[1,]"), None);
        assert_eq!(read("u", "()"), Some(con(0, vec![])));
        assert_eq!(read("b", "True"), Some(con(1, vec![])));
        assert_eq!(read("o", "GT"), Some(con(2, vec![])));
        assert_eq!(
            read("(e i b)", "Right False"),
            Some(con(1, vec![con(0, vec![])]))
        );
    }

    #[test]
    fn application_needs_parentheses_as_an_argument() {
        let nested = "(m (m i))";
        assert_eq!(
            read(nested, "Just (Just 1)"),
            Some(con(1, vec![con(1, vec![Value::Int(1)])]))
        );
        assert_eq!(read(nested, "Just Just 1"), None);
        assert_eq!(
            read(nested, "Just Nothing"),
            Some(con(1, vec![con(0, vec![])]))
        );
        assert_eq!(
            read("(m i)", "Just (-1)"),
            Some(con(1, vec![Value::Int(-1)]))
        );
        assert_eq!(read("(m i)", "Just -1"), None);
        assert_eq!(
            read("(m i)", "((Just 1))"),
            Some(con(1, vec![Value::Int(1)]))
        );
    }

    #[test]
    fn reads_data_types() {
        assert_eq!(read(COLOR, "Blue"), Some(con(2, vec![])));
        assert_eq!(read(COLOR, "Blueish"), None);
        assert_eq!(read(COLOR, "blue"), None);
        assert_eq!(
            read(PERSON, "P {name = \"a\", age = 3}"),
            Some(con(0, vec![Value::Str("a".into()), Value::Int(3)]))
        );
        assert_eq!(
            read(PERSON, "P{name=\"a\",age=3}"),
            read(PERSON, "P {name = \"a\", age = 3}")
        );
        assert_eq!(read(PERSON, "P {age = 3, name = \"a\"}"), None);
        assert_eq!(read(PERSON, "P \"a\" 3"), None);
        let leaf = || con(0, vec![]);
        assert_eq!(
            read(TREE, "Node (Node Leaf 1 Leaf) 2 Leaf"),
            Some(con(
                1,
                vec![
                    con(1, vec![leaf(), Value::Int(1), leaf()]),
                    Value::Int(2),
                    leaf()
                ]
            ))
        );
        assert_eq!(read(TREE, "Node Node Leaf 1 Leaf 2 Leaf"), None);
    }

    #[test]
    fn record_is_an_argument_without_parentheses() {
        let description = "(m @0) (D (C P 0 R age i))";
        assert_eq!(
            read(description, "Just P {age = 3}"),
            Some(con(1, vec![con(0, vec![Value::Int(3)])]))
        );
    }

    #[test]
    fn newtype_is_its_field() {
        assert_eq!(read("@0 (N (C Age 0 P i))", "Age 3"), Some(Value::Int(3)));
        assert_eq!(
            read("@0 (N (C Age 0 R unAge i))", "Age {unAge = 3}"),
            Some(Value::Int(3))
        );
    }

    #[test]
    fn prefix_reports_what_it_consumed() {
        let description = parse_description("i").unwrap();
        let chars: Vec<char> = " 12 rest".chars().collect();
        assert_eq!(read_prefix(&description, &chars), Some((Value::Int(12), 3)));
        assert_eq!(read_prefix(&description, &['x']), None);
    }
}
