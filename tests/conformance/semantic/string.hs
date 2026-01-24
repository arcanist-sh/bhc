-- Test: string
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Phase 3 - Base Library

{-# HASKELL_EDITION 2026 #-}
{-# LANGUAGE OverloadedStrings #-}

module StringTest where

import BHC.Data.String

-- ================================================================
-- IsString Tests
-- ================================================================

testIsStringInstance :: Bool
testIsStringInstance =
    let s :: String
        s = fromString "hello"
    in s == "hello"
-- Result: True

-- ================================================================
-- Lines/Words Tests
-- ================================================================

testLines :: Bool
testLines =
    lines "hello\nworld\n" == ["hello", "world"]
-- Result: True

testLinesEmpty :: Bool
testLinesEmpty =
    lines "" == []
-- Result: True

testLinesSingleLine :: Bool
testLinesSingleLine =
    lines "hello" == ["hello"]
-- Result: True

testLinesMultiple :: Bool
testLinesMultiple =
    lines "a\nb\nc\nd" == ["a", "b", "c", "d"]
-- Result: True

testWords :: Bool
testWords =
    words "hello world" == ["hello", "world"]
-- Result: True

testWordsEmpty :: Bool
testWordsEmpty =
    words "" == []
-- Result: True

testWordsExtraSpaces :: Bool
testWordsExtraSpaces =
    words "  hello   world  " == ["hello", "world"]
-- Result: True

testWordsSingleWord :: Bool
testWordsSingleWord =
    words "hello" == ["hello"]
-- Result: True

testUnlines :: Bool
testUnlines =
    unlines ["hello", "world"] == "hello\nworld\n"
-- Result: True

testUnlinesEmpty :: Bool
testUnlinesEmpty =
    unlines [] == ""
-- Result: True

testUnwords :: Bool
testUnwords =
    unwords ["hello", "world"] == "hello world"
-- Result: True

testUnwordsEmpty :: Bool
testUnwordsEmpty =
    unwords [] == ""
-- Result: True

testUnwordsSingle :: Bool
testUnwordsSingle =
    unwords ["hello"] == "hello"
-- Result: True

-- ================================================================
-- Strip Tests
-- ================================================================

testStrip :: Bool
testStrip =
    strip "  hello world  " == "hello world"
-- Result: True

testStripEmpty :: Bool
testStripEmpty =
    strip "" == ""
-- Result: True

testStripNoWhitespace :: Bool
testStripNoWhitespace =
    strip "hello" == "hello"
-- Result: True

testStripAllWhitespace :: Bool
testStripAllWhitespace =
    strip "   " == ""
-- Result: True

testStripStart :: Bool
testStripStart =
    stripStart "  hello" == "hello" &&
    stripStart "hello  " == "hello  "
-- Result: True

testStripEnd :: Bool
testStripEnd =
    stripEnd "hello  " == "hello" &&
    stripEnd "  hello" == "  hello"
-- Result: True

testStripTabs :: Bool
testStripTabs =
    strip "\t\nhello\t\n" == "hello"
-- Result: True

-- ================================================================
-- SplitOn Tests
-- ================================================================

testSplitOn :: Bool
testSplitOn =
    splitOn "," "a,b,c" == ["a", "b", "c"]
-- Result: True

testSplitOnEmpty :: Bool
testSplitOnEmpty =
    splitOn "," "" == [""]
-- Result: True

testSplitOnNotFound :: Bool
testSplitOnNotFound =
    splitOn "," "abc" == ["abc"]
-- Result: True

testSplitOnMultiChar :: Bool
testSplitOnMultiChar =
    splitOn "::" "a::b::c" == ["a", "b", "c"]
-- Result: True

testSplitOnConsecutive :: Bool
testSplitOnConsecutive =
    splitOn "," ",," == ["", "", ""]
-- Result: True

testSplitOnStart :: Bool
testSplitOnStart =
    splitOn "," ",a,b" == ["", "a", "b"]
-- Result: True

testSplitOnEnd :: Bool
testSplitOnEnd =
    splitOn "," "a,b," == ["a", "b", ""]
-- Result: True

-- ================================================================
-- Replace Tests
-- ================================================================

testReplace :: Bool
testReplace =
    replace "world" "there" "hello world" == "hello there"
-- Result: True

testReplaceMultiple :: Bool
testReplaceMultiple =
    replace "o" "0" "hello world" == "hell0 w0rld"
-- Result: True

testReplaceNotFound :: Bool
testReplaceNotFound =
    replace "x" "y" "hello" == "hello"
-- Result: True

testReplaceEmpty :: Bool
testReplaceEmpty =
    replace "a" "b" "" == ""
-- Result: True

testReplaceWithEmpty :: Bool
testReplaceWithEmpty =
    replace "l" "" "hello" == "heo"
-- Result: True

testReplaceWithLonger :: Bool
testReplaceWithLonger =
    replace "o" "oo" "hello" == "helloo"
-- Result: True

-- ================================================================
-- Predicates Tests
-- ================================================================

testIsSpace :: Bool
testIsSpace =
    isSpace ' ' && isSpace '\t' && isSpace '\n' && not (isSpace 'a')
-- Result: True

testIsAlpha :: Bool
testIsAlpha =
    isAlpha 'a' && isAlpha 'Z' && not (isAlpha '1') && not (isAlpha ' ')
-- Result: True

testIsAlphaNum :: Bool
testIsAlphaNum =
    isAlphaNum 'a' && isAlphaNum '5' && not (isAlphaNum ' ')
-- Result: True

testIsDigit :: Bool
testIsDigit =
    isDigit '0' && isDigit '9' && not (isDigit 'a')
-- Result: True

testIsLower :: Bool
testIsLower =
    isLower 'a' && isLower 'z' && not (isLower 'A')
-- Result: True

testIsUpper :: Bool
testIsUpper =
    isUpper 'A' && isUpper 'Z' && not (isUpper 'a')
-- Result: True

testIsAscii :: Bool
testIsAscii =
    isAscii 'a' && isAscii '\0' && isAscii '\127'
-- Result: True

testIsPrint :: Bool
testIsPrint =
    isPrint 'a' && isPrint ' ' && not (isPrint '\0')
-- Result: True

-- ================================================================
-- Case Conversion Tests
-- ================================================================

testToLower :: Bool
testToLower =
    toLower 'A' == 'a' && toLower 'Z' == 'z' && toLower 'a' == 'a'
-- Result: True

testToUpper :: Bool
testToUpper =
    toUpper 'a' == 'A' && toUpper 'z' == 'Z' && toUpper 'A' == 'A'
-- Result: True

testToTitle :: Bool
testToTitle =
    toTitle 'a' == 'A' && toTitle 'A' == 'A'
-- Result: True

-- ================================================================
-- Justification Tests
-- ================================================================

testJustifyLeft :: Bool
testJustifyLeft =
    justifyLeft 10 '.' "hello" == "hello....."
-- Result: True

testJustifyLeftNoChange :: Bool
testJustifyLeftNoChange =
    justifyLeft 3 '.' "hello" == "hello"  -- String already longer
-- Result: True

testJustifyRight :: Bool
testJustifyRight =
    justifyRight 10 '.' "hello" == ".....hello"
-- Result: True

testJustifyRightNoChange :: Bool
testJustifyRightNoChange =
    justifyRight 3 '.' "hello" == "hello"
-- Result: True

testCenter :: Bool
testCenter =
    center 11 '.' "hello" == "...hello..."
-- Result: True

testCenterOdd :: Bool
testCenterOdd =
    center 10 '.' "hello" == "..hello..."  -- Extra on right
-- Result: True

testCenterNoChange :: Bool
testCenterNoChange =
    center 3 '.' "hello" == "hello"
-- Result: True

-- ================================================================
-- Escape Tests
-- ================================================================

testEscapeNewline :: Bool
testEscapeNewline =
    escape "hello\nworld" == "hello\\nworld"
-- Result: True

testEscapeTab :: Bool
testEscapeTab =
    escape "tab\there" == "tab\\there"
-- Result: True

testEscapeBackslash :: Bool
testEscapeBackslash =
    escape "back\\slash" == "back\\\\slash"
-- Result: True

testEscapeQuote :: Bool
testEscapeQuote =
    escape "say \"hi\"" == "say \\\"hi\\\""
-- Result: True

testEscapeMultiple :: Bool
testEscapeMultiple =
    escape "a\tb\nc" == "a\\tb\\nc"
-- Result: True

testUnescapeNewline :: Bool
testUnescapeNewline =
    unescape "hello\\nworld" == "hello\nworld"
-- Result: True

testUnescapeTab :: Bool
testUnescapeTab =
    unescape "tab\\there" == "tab\there"
-- Result: True

testUnescapeBackslash :: Bool
testUnescapeBackslash =
    unescape "back\\\\slash" == "back\\slash"
-- Result: True

testUnescapeQuote :: Bool
testUnescapeQuote =
    unescape "say \\\"hi\\\"" == "say \"hi\""
-- Result: True

testUnescapeNoop :: Bool
testUnescapeNoop =
    unescape "hello" == "hello"  -- No escapes
-- Result: True

-- ================================================================
-- Prefix/Suffix/Infix Tests
-- ================================================================

testIsPrefixOf :: Bool
testIsPrefixOf =
    isPrefixOf "hel" "hello" && not (isPrefixOf "wor" "hello")
-- Result: True

testIsSuffixOf :: Bool
testIsSuffixOf =
    isSuffixOf "llo" "hello" && not (isSuffixOf "hel" "hello")
-- Result: True

testIsInfixOf :: Bool
testIsInfixOf =
    isInfixOf "ell" "hello" && not (isInfixOf "xyz" "hello")
-- Result: True

testIsPrefixOfEmpty :: Bool
testIsPrefixOfEmpty =
    isPrefixOf "" "hello"  -- Empty is prefix of everything
-- Result: True

testIsSuffixOfEmpty :: Bool
testIsSuffixOfEmpty =
    isSuffixOf "" "hello"  -- Empty is suffix of everything
-- Result: True

testIsInfixOfEmpty :: Bool
testIsInfixOfEmpty =
    isInfixOf "" "hello"  -- Empty is infix of everything
-- Result: True

-- ================================================================
-- Intercalate/Intersperse Tests
-- ================================================================

testIntercalate :: Bool
testIntercalate =
    intercalate ", " ["a", "b", "c"] == "a, b, c"
-- Result: True

testIntercalateEmpty :: Bool
testIntercalateEmpty =
    intercalate ", " [] == ""
-- Result: True

testIntercalateSingle :: Bool
testIntercalateSingle =
    intercalate ", " ["a"] == "a"
-- Result: True

testIntersperse :: Bool
testIntersperse =
    intersperse ',' "abc" == "a,b,c"
-- Result: True

testIntersperseEmpty :: Bool
testIntersperseEmpty =
    intersperse ',' "" == ""
-- Result: True

-- ================================================================
-- Edge Cases
-- ================================================================

testEmptyString :: Bool
testEmptyString =
    lines "" == [] &&
    words "" == [] &&
    strip "" == "" &&
    escape "" == "" &&
    unescape "" == ""
-- Result: True

testWhitespaceOnly :: Bool
testWhitespaceOnly =
    words "   " == [] &&
    strip "   " == "" &&
    lines "   " == ["   "]
-- Result: True

testSpecialChars :: Bool
testSpecialChars =
    escape "\r\n\t" == "\\r\\n\\t"
-- Result: True

-- ================================================================
-- Property-style Tests
-- ================================================================

-- lines then unlines almost identity (adds trailing newline)
propLinesUnlines :: Bool
propLinesUnlines =
    let s = "a\nb\nc"
    in unlines (lines s) == s ++ "\n"
-- Result: True

-- words then unwords preserves words
propWordsUnwords :: Bool
propWordsUnwords =
    let ws = ["hello", "world"]
    in words (unwords ws) == ws
-- Result: True

-- strip is idempotent
propStripIdempotent :: Bool
propStripIdempotent =
    let s = "  hello  "
    in strip (strip s) == strip s
-- Result: True

-- escape then unescape is identity for printable strings
propEscapeUnescape :: Bool
propEscapeUnescape =
    let s = "hello\nworld\ttab"
    in unescape (escape s) == s
-- Result: True

-- ================================================================
-- Main
-- ================================================================

main :: IO ()
main = do
    -- IsString
    print testIsStringInstance

    -- Lines/Words
    print testLines
    print testLinesEmpty
    print testLinesSingleLine
    print testLinesMultiple
    print testWords
    print testWordsEmpty
    print testWordsExtraSpaces
    print testWordsSingleWord
    print testUnlines
    print testUnlinesEmpty
    print testUnwords
    print testUnwordsEmpty
    print testUnwordsSingle

    -- Strip
    print testStrip
    print testStripEmpty
    print testStripNoWhitespace
    print testStripAllWhitespace
    print testStripStart
    print testStripEnd
    print testStripTabs

    -- SplitOn
    print testSplitOn
    print testSplitOnEmpty
    print testSplitOnNotFound
    print testSplitOnMultiChar
    print testSplitOnConsecutive
    print testSplitOnStart
    print testSplitOnEnd

    -- Replace
    print testReplace
    print testReplaceMultiple
    print testReplaceNotFound
    print testReplaceEmpty
    print testReplaceWithEmpty
    print testReplaceWithLonger

    -- Predicates
    print testIsSpace
    print testIsAlpha
    print testIsAlphaNum
    print testIsDigit
    print testIsLower
    print testIsUpper
    print testIsAscii
    print testIsPrint

    -- Case conversion
    print testToLower
    print testToUpper
    print testToTitle

    -- Justification
    print testJustifyLeft
    print testJustifyLeftNoChange
    print testJustifyRight
    print testJustifyRightNoChange
    print testCenter
    print testCenterOdd
    print testCenterNoChange

    -- Escape
    print testEscapeNewline
    print testEscapeTab
    print testEscapeBackslash
    print testEscapeQuote
    print testEscapeMultiple
    print testUnescapeNewline
    print testUnescapeTab
    print testUnescapeBackslash
    print testUnescapeQuote
    print testUnescapeNoop

    -- Prefix/Suffix/Infix
    print testIsPrefixOf
    print testIsSuffixOf
    print testIsInfixOf
    print testIsPrefixOfEmpty
    print testIsSuffixOfEmpty
    print testIsInfixOfEmpty

    -- Intercalate/Intersperse
    print testIntercalate
    print testIntercalateEmpty
    print testIntercalateSingle
    print testIntersperse
    print testIntersperseEmpty

    -- Edge cases
    print testEmptyString
    print testWhitespaceOnly
    print testSpecialChars

    -- Properties
    print propLinesUnlines
    print propWordsUnwords
    print propStripIdempotent
    print propEscapeUnescape
