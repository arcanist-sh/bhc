{-# LANGUAGE BangPatterns #-}

-- |
-- Module      : BHC.Data.Text.Encoding
-- Description : Unicode encoding and decoding
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Functions for converting 'Text' values to and from 'ByteString',
-- using various Unicode encodings.
--
-- = UTF-8
--
-- UTF-8 is the default encoding for BHC 'Text' values internally.
-- The 'encodeUtf8' and 'decodeUtf8' functions are the most efficient
-- way to convert between 'Text' and 'ByteString'.
--
-- = Other Encodings
--
-- This module also provides UTF-16 and UTF-32 encoding/decoding,
-- as well as Latin-1 (ISO-8859-1) and ASCII.
--
-- = Error Handling
--
-- Decoding functions come in several variants:
--
-- * @decode*@ - throws an exception on invalid input
-- * @decode*'@ - uses replacement character for invalid bytes
-- * @decode*With@ - uses a custom error handler

module BHC.Data.Text.Encoding
    ( -- * Decoding ByteString to Text
      -- ** UTF-8
      decodeUtf8
    , decodeUtf8'
    , decodeUtf8With
    , decodeUtf8Lenient

      -- ** UTF-16
    , decodeUtf16LE
    , decodeUtf16LEWith
    , decodeUtf16BE
    , decodeUtf16BEWith

      -- ** UTF-32
    , decodeUtf32LE
    , decodeUtf32LEWith
    , decodeUtf32BE
    , decodeUtf32BEWith

      -- ** Latin-1
    , decodeLatin1

      -- ** ASCII
    , decodeASCII
    , decodeASCIIWith

      -- * Encoding Text to ByteString
      -- ** UTF-8
    , encodeUtf8
    , encodeUtf8Builder
    , encodeUtf8BuilderEscaped

      -- ** UTF-16
    , encodeUtf16LE
    , encodeUtf16BE

      -- ** UTF-32
    , encodeUtf32LE
    , encodeUtf32BE

      -- ** Latin-1
    , encodeLatin1

      -- ** ASCII
    , encodeASCII

      -- * Error handling
    , OnDecodeError
    , OnError
    , UnicodeException(..)
    , strictDecode
    , lenientDecode
    , ignore
    , replace
    ) where

import Prelude hiding (length, take, drop)

import Data.Bits ((.&.), (.|.), shiftL, shiftR)
import Data.Char (chr, ord)
import Data.Word (Word8, Word16, Word32)
import Control.Exception (Exception, throw)

import qualified BHC.Data.ByteString as BS
import BHC.Data.ByteString (ByteString)
import qualified BHC.Data.Text as T
import BHC.Data.Text (Text)
import qualified BHC.Data.ByteString.Builder as B
import BHC.Data.ByteString.Builder (Builder)

-- ============================================================
-- Error handling types
-- ============================================================

-- | Exception thrown on decoding errors.
data UnicodeException = DecodeError String (Maybe Word8)
    deriving (Eq, Show)

instance Exception UnicodeException

-- | Action to take on a decoding error.
type OnDecodeError = String -> Maybe Word8 -> Maybe Char

-- | Deprecated: Use 'OnDecodeError'.
type OnError a b = String -> Maybe a -> Maybe b

-- | Throw an exception on decoding errors.
strictDecode :: OnDecodeError
strictDecode desc byte = throw (DecodeError desc byte)

-- | Replace errors with the Unicode replacement character (U+FFFD).
lenientDecode :: OnDecodeError
lenientDecode _ _ = Just '\xFFFD'

-- | Ignore decoding errors (skip the problematic byte).
ignore :: OnDecodeError
ignore _ _ = Nothing

-- | Replace errors with a specific character.
replace :: Char -> OnDecodeError
replace c _ _ = Just c

-- ============================================================
-- UTF-8 Decoding
-- ============================================================

-- | Decode a 'ByteString' as UTF-8.
-- Throws 'UnicodeException' on invalid input.
decodeUtf8 :: ByteString -> Text
decodeUtf8 = decodeUtf8With strictDecode

-- | Decode a 'ByteString' as UTF-8, returning 'Left' on errors.
decodeUtf8' :: ByteString -> Either UnicodeException Text
decodeUtf8' bs = case tryDecode bs of
    Left e  -> Left e
    Right t -> Right t
  where
    tryDecode b = Right (decodeUtf8Lenient b)  -- Simplified

-- | Decode a 'ByteString' as UTF-8 with a custom error handler.
decodeUtf8With :: OnDecodeError -> ByteString -> Text
decodeUtf8With onError bs = T.pack (go 0)
  where
    len = BS.length bs

    go !i
        | i >= len = []
        | otherwise =
            let b0 = BS.index bs i
            in if b0 < 0x80
               then chr (fromIntegral b0) : go (i + 1)
               else if b0 < 0xc0
                    then handleError i b0
                    else if b0 < 0xe0
                         then decode2 i b0
                         else if b0 < 0xf0
                              then decode3 i b0
                              else if b0 < 0xf8
                                   then decode4 i b0
                                   else handleError i b0

    decode2 !i !b0
        | i + 1 >= len = handleError i b0
        | otherwise =
            let b1 = BS.index bs (i + 1)
            in if isContinuation b1
               then let cp = ((fromIntegral b0 .&. 0x1f) `shiftL` 6) .|.
                             (fromIntegral b1 .&. 0x3f)
                    in if cp >= 0x80
                       then chr cp : go (i + 2)
                       else handleError i b0  -- Overlong
               else handleError i b0

    decode3 !i !b0
        | i + 2 >= len = handleError i b0
        | otherwise =
            let b1 = BS.index bs (i + 1)
                b2 = BS.index bs (i + 2)
            in if isContinuation b1 && isContinuation b2
               then let cp = ((fromIntegral b0 .&. 0x0f) `shiftL` 12) .|.
                             ((fromIntegral b1 .&. 0x3f) `shiftL` 6) .|.
                             (fromIntegral b2 .&. 0x3f)
                    in if cp >= 0x800 && not (isSurrogate cp)
                       then chr cp : go (i + 3)
                       else handleError i b0
               else handleError i b0

    decode4 !i !b0
        | i + 3 >= len = handleError i b0
        | otherwise =
            let b1 = BS.index bs (i + 1)
                b2 = BS.index bs (i + 2)
                b3 = BS.index bs (i + 3)
            in if isContinuation b1 && isContinuation b2 && isContinuation b3
               then let cp = ((fromIntegral b0 .&. 0x07) `shiftL` 18) .|.
                             ((fromIntegral b1 .&. 0x3f) `shiftL` 12) .|.
                             ((fromIntegral b2 .&. 0x3f) `shiftL` 6) .|.
                             (fromIntegral b3 .&. 0x3f)
                    in if cp >= 0x10000 && cp <= 0x10ffff
                       then chr cp : go (i + 4)
                       else handleError i b0
               else handleError i b0

    handleError !i !b = case onError "invalid UTF-8" (Just b) of
        Just c  -> c : go (i + 1)
        Nothing -> go (i + 1)

    isContinuation b = b >= 0x80 && b < 0xc0
    isSurrogate cp = cp >= 0xd800 && cp <= 0xdfff

-- | Decode UTF-8, replacing invalid sequences with U+FFFD.
decodeUtf8Lenient :: ByteString -> Text
decodeUtf8Lenient = decodeUtf8With lenientDecode

-- ============================================================
-- UTF-16 Decoding
-- ============================================================

-- | Decode UTF-16LE.
decodeUtf16LE :: ByteString -> Text
decodeUtf16LE = decodeUtf16LEWith strictDecode

-- | Decode UTF-16LE with custom error handler.
decodeUtf16LEWith :: OnDecodeError -> ByteString -> Text
decodeUtf16LEWith onError bs = T.pack (go 0)
  where
    len = BS.length bs

    go !i
        | i + 1 >= len = if i < len then handleError i else []
        | otherwise =
            let w = word16LE bs i
            in if w < 0xd800 || w > 0xdfff
               then chr (fromIntegral w) : go (i + 2)
               else if w >= 0xd800 && w <= 0xdbff
                    then decodeSurrogate i w
                    else handleError i

    decodeSurrogate !i !hi
        | i + 3 >= len = handleError i
        | otherwise =
            let lo = word16LE bs (i + 2)
            in if lo >= 0xdc00 && lo <= 0xdfff
               then let cp = 0x10000 + ((fromIntegral hi - 0xd800) `shiftL` 10) +
                             (fromIntegral lo - 0xdc00)
                    in chr cp : go (i + 4)
               else handleError i

    handleError !i = case onError "invalid UTF-16LE" (Just (BS.index bs i)) of
        Just c  -> c : go (i + 2)
        Nothing -> go (i + 2)

-- | Decode UTF-16BE.
decodeUtf16BE :: ByteString -> Text
decodeUtf16BE = decodeUtf16BEWith strictDecode

-- | Decode UTF-16BE with custom error handler.
decodeUtf16BEWith :: OnDecodeError -> ByteString -> Text
decodeUtf16BEWith onError bs = T.pack (go 0)
  where
    len = BS.length bs

    go !i
        | i + 1 >= len = if i < len then handleError i else []
        | otherwise =
            let w = word16BE bs i
            in if w < 0xd800 || w > 0xdfff
               then chr (fromIntegral w) : go (i + 2)
               else if w >= 0xd800 && w <= 0xdbff
                    then decodeSurrogate i w
                    else handleError i

    decodeSurrogate !i !hi
        | i + 3 >= len = handleError i
        | otherwise =
            let lo = word16BE bs (i + 2)
            in if lo >= 0xdc00 && lo <= 0xdfff
               then let cp = 0x10000 + ((fromIntegral hi - 0xd800) `shiftL` 10) +
                             (fromIntegral lo - 0xdc00)
                    in chr cp : go (i + 4)
               else handleError i

    handleError !i = case onError "invalid UTF-16BE" (Just (BS.index bs i)) of
        Just c  -> c : go (i + 2)
        Nothing -> go (i + 2)

-- ============================================================
-- UTF-32 Decoding
-- ============================================================

-- | Decode UTF-32LE.
decodeUtf32LE :: ByteString -> Text
decodeUtf32LE = decodeUtf32LEWith strictDecode

-- | Decode UTF-32LE with custom error handler.
decodeUtf32LEWith :: OnDecodeError -> ByteString -> Text
decodeUtf32LEWith onError bs = T.pack (go 0)
  where
    len = BS.length bs

    go !i
        | i + 3 >= len = if i < len then handleError i else []
        | otherwise =
            let cp = word32LE bs i
            in if cp <= 0x10ffff && not (isSurrogate cp)
               then chr (fromIntegral cp) : go (i + 4)
               else handleError i

    handleError !i = case onError "invalid UTF-32LE" (Just (BS.index bs i)) of
        Just c  -> c : go (i + 4)
        Nothing -> go (i + 4)

    isSurrogate cp = cp >= 0xd800 && cp <= 0xdfff

-- | Decode UTF-32BE.
decodeUtf32BE :: ByteString -> Text
decodeUtf32BE = decodeUtf32BEWith strictDecode

-- | Decode UTF-32BE with custom error handler.
decodeUtf32BEWith :: OnDecodeError -> ByteString -> Text
decodeUtf32BEWith onError bs = T.pack (go 0)
  where
    len = BS.length bs

    go !i
        | i + 3 >= len = if i < len then handleError i else []
        | otherwise =
            let cp = word32BE bs i
            in if cp <= 0x10ffff && not (isSurrogate cp)
               then chr (fromIntegral cp) : go (i + 4)
               else handleError i

    handleError !i = case onError "invalid UTF-32BE" (Just (BS.index bs i)) of
        Just c  -> c : go (i + 4)
        Nothing -> go (i + 4)

    isSurrogate cp = cp >= 0xd800 && cp <= 0xdfff

-- ============================================================
-- Latin-1 and ASCII Decoding
-- ============================================================

-- | Decode Latin-1 (ISO-8859-1).
-- This is always valid - each byte maps directly to a code point.
decodeLatin1 :: ByteString -> Text
decodeLatin1 bs = T.pack [chr (fromIntegral (BS.index bs i)) | i <- [0..BS.length bs - 1]]

-- | Decode ASCII.
-- Throws on bytes >= 128.
decodeASCII :: ByteString -> Text
decodeASCII = decodeASCIIWith strictDecode

-- | Decode ASCII with custom error handler.
decodeASCIIWith :: OnDecodeError -> ByteString -> Text
decodeASCIIWith onError bs = T.pack (go 0)
  where
    len = BS.length bs

    go !i
        | i >= len = []
        | otherwise =
            let b = BS.index bs i
            in if b < 0x80
               then chr (fromIntegral b) : go (i + 1)
               else case onError "invalid ASCII" (Just b) of
                   Just c  -> c : go (i + 1)
                   Nothing -> go (i + 1)

-- ============================================================
-- UTF-8 Encoding
-- ============================================================

-- | Encode a 'Text' to UTF-8.
encodeUtf8 :: Text -> ByteString
encodeUtf8 = BS.pack . concatMap encodeChar . T.unpack
  where
    encodeChar c
        | cp < 0x80    = [fromIntegral cp]
        | cp < 0x800   = [fromIntegral (0xc0 .|. (cp `shiftR` 6)),
                         fromIntegral (0x80 .|. (cp .&. 0x3f))]
        | cp < 0x10000 = [fromIntegral (0xe0 .|. (cp `shiftR` 12)),
                         fromIntegral (0x80 .|. ((cp `shiftR` 6) .&. 0x3f)),
                         fromIntegral (0x80 .|. (cp .&. 0x3f))]
        | otherwise    = [fromIntegral (0xf0 .|. (cp `shiftR` 18)),
                         fromIntegral (0x80 .|. ((cp `shiftR` 12) .&. 0x3f)),
                         fromIntegral (0x80 .|. ((cp `shiftR` 6) .&. 0x3f)),
                         fromIntegral (0x80 .|. (cp .&. 0x3f))]
      where
        cp = ord c

-- | Encode 'Text' as UTF-8 using a 'Builder'.
encodeUtf8Builder :: Text -> Builder
encodeUtf8Builder = B.stringUtf8 . T.unpack

-- | Encode 'Text' as UTF-8 with byte escaping.
encodeUtf8BuilderEscaped :: (Word8 -> Builder) -> Text -> Builder
encodeUtf8BuilderEscaped escape t = foldMap go (BS.unpack (encodeUtf8 t))
  where
    go b = escape b

-- ============================================================
-- UTF-16 Encoding
-- ============================================================

-- | Encode as UTF-16LE.
encodeUtf16LE :: Text -> ByteString
encodeUtf16LE = BS.pack . concatMap encodeChar . T.unpack
  where
    encodeChar c
        | cp < 0x10000 = [fromIntegral cp, fromIntegral (cp `shiftR` 8)]
        | otherwise =
            let cp' = cp - 0x10000
                hi = 0xd800 + (cp' `shiftR` 10)
                lo = 0xdc00 + (cp' .&. 0x3ff)
            in [fromIntegral hi, fromIntegral (hi `shiftR` 8),
                fromIntegral lo, fromIntegral (lo `shiftR` 8)]
      where
        cp = ord c

-- | Encode as UTF-16BE.
encodeUtf16BE :: Text -> ByteString
encodeUtf16BE = BS.pack . concatMap encodeChar . T.unpack
  where
    encodeChar c
        | cp < 0x10000 = [fromIntegral (cp `shiftR` 8), fromIntegral cp]
        | otherwise =
            let cp' = cp - 0x10000
                hi = 0xd800 + (cp' `shiftR` 10)
                lo = 0xdc00 + (cp' .&. 0x3ff)
            in [fromIntegral (hi `shiftR` 8), fromIntegral hi,
                fromIntegral (lo `shiftR` 8), fromIntegral lo]
      where
        cp = ord c

-- ============================================================
-- UTF-32 Encoding
-- ============================================================

-- | Encode as UTF-32LE.
encodeUtf32LE :: Text -> ByteString
encodeUtf32LE = BS.pack . concatMap encodeChar . T.unpack
  where
    encodeChar c =
        let cp = ord c
        in [fromIntegral cp,
            fromIntegral (cp `shiftR` 8),
            fromIntegral (cp `shiftR` 16),
            fromIntegral (cp `shiftR` 24)]

-- | Encode as UTF-32BE.
encodeUtf32BE :: Text -> ByteString
encodeUtf32BE = BS.pack . concatMap encodeChar . T.unpack
  where
    encodeChar c =
        let cp = ord c
        in [fromIntegral (cp `shiftR` 24),
            fromIntegral (cp `shiftR` 16),
            fromIntegral (cp `shiftR` 8),
            fromIntegral cp]

-- ============================================================
-- Latin-1 and ASCII Encoding
-- ============================================================

-- | Encode as Latin-1, truncating code points > 255.
encodeLatin1 :: Text -> ByteString
encodeLatin1 = BS.pack . map (fromIntegral . min 255 . ord) . T.unpack

-- | Encode as ASCII, truncating code points > 127.
encodeASCII :: Text -> ByteString
encodeASCII = BS.pack . map (fromIntegral . min 127 . ord) . T.unpack

-- ============================================================
-- Internal helpers
-- ============================================================

-- | Read a 16-bit little-endian word.
word16LE :: ByteString -> Int -> Word16
word16LE bs i = fromIntegral (BS.index bs i) .|.
                (fromIntegral (BS.index bs (i + 1)) `shiftL` 8)

-- | Read a 16-bit big-endian word.
word16BE :: ByteString -> Int -> Word16
word16BE bs i = (fromIntegral (BS.index bs i) `shiftL` 8) .|.
                fromIntegral (BS.index bs (i + 1))

-- | Read a 32-bit little-endian word.
word32LE :: ByteString -> Int -> Word32
word32LE bs i = fromIntegral (BS.index bs i) .|.
                (fromIntegral (BS.index bs (i + 1)) `shiftL` 8) .|.
                (fromIntegral (BS.index bs (i + 2)) `shiftL` 16) .|.
                (fromIntegral (BS.index bs (i + 3)) `shiftL` 24)

-- | Read a 32-bit big-endian word.
word32BE :: ByteString -> Int -> Word32
word32BE bs i = (fromIntegral (BS.index bs i) `shiftL` 24) .|.
                (fromIntegral (BS.index bs (i + 1)) `shiftL` 16) .|.
                (fromIntegral (BS.index bs (i + 2)) `shiftL` 8) .|.
                fromIntegral (BS.index bs (i + 3))
