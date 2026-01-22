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

-- Classification
isControl :: Char -> Bool
isControl c = c < ' ' || c >= '\DEL' && c <= '\x9f'

isSpace :: Char -> Bool
isSpace c = c `elem` " \t\n\r\f\v"

isLower :: Char -> Bool
isLower c = c >= 'a' && c <= 'z'

isUpper :: Char -> Bool
isUpper c = c >= 'A' && c <= 'Z'

isAlpha :: Char -> Bool
isAlpha c = isLower c || isUpper c

isAlphaNum :: Char -> Bool
isAlphaNum c = isAlpha c || isDigit c

isPrint :: Char -> Bool
isPrint c = c >= ' ' && c <= '~'

isDigit :: Char -> Bool
isDigit c = c >= '0' && c <= '9'

isOctDigit :: Char -> Bool
isOctDigit c = c >= '0' && c <= '7'

isHexDigit :: Char -> Bool
isHexDigit c = isDigit c || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')

isLetter :: Char -> Bool
isLetter = isAlpha

isMark :: Char -> Bool
isMark _ = False  -- Simplified

isNumber :: Char -> Bool
isNumber = isDigit

isPunctuation :: Char -> Bool
isPunctuation c = c `elem` "!\"#%&'()*,-./:;?@[\\]_{}"

isSymbol :: Char -> Bool
isSymbol c = c `elem` "$+<=>^`|~"

isSeparator :: Char -> Bool
isSeparator = isSpace

isAscii :: Char -> Bool
isAscii c = ord c < 128

isLatin1 :: Char -> Bool
isLatin1 c = ord c < 256

isAsciiUpper :: Char -> Bool
isAsciiUpper c = c >= 'A' && c <= 'Z'

isAsciiLower :: Char -> Bool
isAsciiLower c = c >= 'a' && c <= 'z'

-- Case conversion
toUpper :: Char -> Char
toUpper c | isLower c = chr (ord c - 32)
          | otherwise = c

toLower :: Char -> Char
toLower c | isUpper c = chr (ord c + 32)
          | otherwise = c

toTitle :: Char -> Char
toTitle = toUpper

-- Digit conversion
digitToInt :: Char -> Int
digitToInt c
    | isDigit c    = ord c - ord '0'
    | c >= 'a' && c <= 'f' = ord c - ord 'a' + 10
    | c >= 'A' && c <= 'F' = ord c - ord 'A' + 10
    | otherwise    = error "digitToInt: not a digit"

intToDigit :: Int -> Char
intToDigit n
    | n >= 0 && n <= 9   = chr (ord '0' + n)
    | n >= 10 && n <= 15 = chr (ord 'a' + n - 10)
    | otherwise = error "intToDigit: not a digit"

-- Ord/Chr
foreign import ccall "bhc_char_ord" ord :: Char -> Int
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
