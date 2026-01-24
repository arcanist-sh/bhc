-- |
-- Module      : BHC.Data.Char
-- Description : Character classification and conversion
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Data.Char (
    -- * Character classification
    isControl, isSpace, isLower, isUpper, isAlpha,
    isAlphaNum, isPrint, isDigit, isOctDigit, isHexDigit,
    isLetter, isMark, isNumber, isPunctuation, isSymbol,
    isSeparator, isAscii, isLatin1, isAsciiUpper, isAsciiLower,
    
    -- * Case conversion
    toUpper, toLower, toTitle,
    
    -- * Single digit
    digitToInt, intToDigit,
    
    -- * Numeric representation
    ord, chr,
    
    -- * String conversion
    showLitChar, readLitChar, lexLitChar,
) where

import BHC.Prelude hiding (Char)

-- | Character type
type Char = GHC.Types.Char

-- ------------------------------------------------------------
-- Character Classification
-- ------------------------------------------------------------

-- | /O(1)/. Is the character a control character?
--
-- >>> isControl '\n'
-- True
-- >>> isControl 'a'
-- False
isControl :: Char -> Bool
isControl c = c < ' ' || c >= '\DEL' && c <= '\x9f'

-- | /O(1)/. Is the character a whitespace character?
-- Includes space, tab, newline, carriage return, form feed, vertical tab.
--
-- >>> isSpace ' '
-- True
-- >>> isSpace 'a'
-- False
isSpace :: Char -> Bool
isSpace c = c `elem` " \t\n\r\f\v"

-- | /O(1)/. Is the character a lowercase ASCII letter?
--
-- >>> isLower 'a'
-- True
-- >>> isLower 'A'
-- False
isLower :: Char -> Bool
isLower c = c >= 'a' && c <= 'z'

-- | /O(1)/. Is the character an uppercase ASCII letter?
--
-- >>> isUpper 'A'
-- True
-- >>> isUpper 'a'
-- False
isUpper :: Char -> Bool
isUpper c = c >= 'A' && c <= 'Z'

-- | /O(1)/. Is the character an ASCII letter?
--
-- >>> isAlpha 'x'
-- True
-- >>> isAlpha '5'
-- False
isAlpha :: Char -> Bool
isAlpha c = isLower c || isUpper c

-- | /O(1)/. Is the character an ASCII letter or digit?
--
-- >>> isAlphaNum 'a'
-- True
-- >>> isAlphaNum '5'
-- True
-- >>> isAlphaNum '!'
-- False
isAlphaNum :: Char -> Bool
isAlphaNum c = isAlpha c || isDigit c

-- | /O(1)/. Is the character printable ASCII?
--
-- >>> isPrint 'a'
-- True
-- >>> isPrint '\n'
-- False
isPrint :: Char -> Bool
isPrint c = c >= ' ' && c <= '~'

-- | /O(1)/. Is the character an ASCII decimal digit (0-9)?
--
-- >>> isDigit '5'
-- True
-- >>> isDigit 'a'
-- False
isDigit :: Char -> Bool
isDigit c = c >= '0' && c <= '9'

-- | /O(1)/. Is the character an ASCII octal digit (0-7)?
--
-- >>> isOctDigit '7'
-- True
-- >>> isOctDigit '8'
-- False
isOctDigit :: Char -> Bool
isOctDigit c = c >= '0' && c <= '7'

-- | /O(1)/. Is the character an ASCII hexadecimal digit (0-9, a-f, A-F)?
--
-- >>> isHexDigit 'f'
-- True
-- >>> isHexDigit 'g'
-- False
isHexDigit :: Char -> Bool
isHexDigit c = isDigit c || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')

-- | /O(1)/. Alias for 'isAlpha'.
isLetter :: Char -> Bool
isLetter = isAlpha

-- | /O(1)/. Is the character a Unicode mark? (Simplified: always False)
isMark :: Char -> Bool
isMark _ = False  -- Simplified

-- | /O(1)/. Alias for 'isDigit'.
isNumber :: Char -> Bool
isNumber = isDigit

-- | /O(1)/. Is the character ASCII punctuation?
--
-- >>> isPunctuation '!'
-- True
-- >>> isPunctuation 'a'
-- False
isPunctuation :: Char -> Bool
isPunctuation c = c `elem` "!\"#%&'()*,-./:;?@[\\]_{}"

-- | /O(1)/. Is the character an ASCII symbol?
--
-- >>> isSymbol '+'
-- True
-- >>> isSymbol 'a'
-- False
isSymbol :: Char -> Bool
isSymbol c = c `elem` "$+<=>^`|~"

-- | /O(1)/. Alias for 'isSpace'.
isSeparator :: Char -> Bool
isSeparator = isSpace

-- | /O(1)/. Is the character in the ASCII range (0-127)?
--
-- >>> isAscii 'a'
-- True
isAscii :: Char -> Bool
isAscii c = ord c < 128

-- | /O(1)/. Is the character in the Latin-1 range (0-255)?
--
-- >>> isLatin1 'a'
-- True
isLatin1 :: Char -> Bool
isLatin1 c = ord c < 256

-- | /O(1)/. Is the character an ASCII uppercase letter?
--
-- >>> isAsciiUpper 'A'
-- True
isAsciiUpper :: Char -> Bool
isAsciiUpper c = c >= 'A' && c <= 'Z'

-- | /O(1)/. Is the character an ASCII lowercase letter?
--
-- >>> isAsciiLower 'a'
-- True
isAsciiLower :: Char -> Bool
isAsciiLower c = c >= 'a' && c <= 'z'

-- ------------------------------------------------------------
-- Case Conversion
-- ------------------------------------------------------------

-- | /O(1)/. Convert to uppercase (ASCII only).
--
-- >>> toUpper 'a'
-- 'A'
-- >>> toUpper 'A'
-- 'A'
toUpper :: Char -> Char
toUpper c | isLower c = chr (ord c - 32)
          | otherwise = c

-- | /O(1)/. Convert to lowercase (ASCII only).
--
-- >>> toLower 'A'
-- 'a'
-- >>> toLower 'a'
-- 'a'
toLower :: Char -> Char
toLower c | isUpper c = chr (ord c + 32)
          | otherwise = c

-- | /O(1)/. Convert to title case (same as 'toUpper' for ASCII).
--
-- >>> toTitle 'a'
-- 'A'
toTitle :: Char -> Char
toTitle = toUpper

-- ------------------------------------------------------------
-- Digit Conversion
-- ------------------------------------------------------------

-- | /O(1)/. Convert a hexadecimal digit to its integer value.
--
-- >>> digitToInt '5'
-- 5
-- >>> digitToInt 'a'
-- 10
-- >>> digitToInt 'F'
-- 15
--
-- __Warning__: Partial function. Throws an error for non-hex digits.
digitToInt :: Char -> Int
digitToInt c
    | isDigit c    = ord c - ord '0'
    | c >= 'a' && c <= 'f' = ord c - ord 'a' + 10
    | c >= 'A' && c <= 'F' = ord c - ord 'A' + 10
    | otherwise    = error "digitToInt: not a digit"

-- | /O(1)/. Convert an integer (0-15) to a hexadecimal digit.
--
-- >>> intToDigit 5
-- '5'
-- >>> intToDigit 10
-- 'a'
--
-- __Warning__: Partial function. Throws an error for values outside 0-15.
intToDigit :: Int -> Char
intToDigit n
    | n >= 0 && n <= 9   = chr (ord '0' + n)
    | n >= 10 && n <= 15 = chr (ord 'a' + n - 10)
    | otherwise = error "intToDigit: not a digit"

-- ------------------------------------------------------------
-- Numeric Representation
-- ------------------------------------------------------------

-- | /O(1)/. Convert a character to its Unicode code point.
--
-- >>> ord 'A'
-- 65
-- >>> ord 'a'
-- 97
foreign import ccall "bhc_char_ord" ord :: Char -> Int

-- | /O(1)/. Convert a Unicode code point to a character.
--
-- >>> chr 65
-- 'A'
-- >>> chr 97
-- 'a'
foreign import ccall "bhc_char_chr" chr :: Int -> Char

-- String representation
showLitChar :: Char -> ShowS
showLitChar c s | c > '\DEL' = showChar '\\' (shows (ord c) s)
showLitChar '\DEL' s = showString "\\DEL" s
showLitChar '\\' s   = showString "\\\\" s
showLitChar '\'' s   = showString "\\'" s
showLitChar '\n' s   = showString "\\n" s
showLitChar '\t' s   = showString "\\t" s
showLitChar '\r' s   = showString "\\r" s
showLitChar c s | isPrint c = c : s
                | otherwise = showString "\\x" (showHex (ord c) s)
  where showHex n = showString (intToHex n)
        intToHex n = if n < 16 then [intToDigit n]
                     else intToHex (n `div` 16) ++ [intToDigit (n `mod` 16)]

readLitChar :: ReadS Char
readLitChar ('\\':s) = readEsc s
readLitChar (c:s)    = [(c, s)]
readLitChar []       = []

readEsc :: ReadS Char
readEsc ('n':s)  = [('\n', s)]
readEsc ('t':s)  = [('\t', s)]
readEsc ('r':s)  = [('\r', s)]
readEsc ('\\':s) = [('\\', s)]
readEsc ('\'':s) = [('\'', s)]
readEsc _        = []

lexLitChar :: ReadS String
lexLitChar ('\\':s) = [('\\':take 1 s, drop 1 s)]
lexLitChar (c:s)    = [([c], s)]
lexLitChar []       = []
