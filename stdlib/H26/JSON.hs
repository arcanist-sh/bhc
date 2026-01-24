-- |
-- Module      : H26.JSON
-- Description : Minimal JSON API
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- JSON encoding and decoding for web service interoperability.
--
-- = Overview
--
-- This module provides type-safe JSON parsing and generation using:
--
-- * 'Value' type for JSON values
-- * 'ToJSON' class for encoding Haskell types
-- * 'FromJSON' class for decoding JSON
-- * Combinators for building parsers
--
-- = Quick Start
--
-- @
-- import H26.JSON
--
-- -- Decode JSON
-- case decode \"{\\\"name\\\": \\\"Alice\\\", \\\"age\\\": 30}\" of
--     Just val -> print (val .: \"name\")
--     Nothing  -> putStrLn \"Parse error\"
--
-- -- Build JSON
-- let person = object
--         [ \"name\" .= \"Bob\"
--         , \"age\" .= (25 :: Int)
--         , \"active\" .= True
--         ]
-- putStrLn (encodePretty person)
-- @
--
-- = Custom Types
--
-- Derive or implement 'ToJSON'\/'FromJSON' for your types:
--
-- @
-- data User = User { name :: Text, age :: Int }
--
-- instance ToJSON User where
--     toJSON (User n a) = object [\"name\" .= n, \"age\" .= a]
--
-- instance FromJSON User where
--     parseJSON = withObject \"User\" $ \\o ->
--         User \<$\> o .: \"name\" \<*\> o .: \"age\"
-- @
--
-- = Operators
--
-- * '.:' — Required field lookup (fails if missing)
-- * '.:?' — Optional field lookup (returns 'Maybe')
-- * '.!=' — Provide default for optional field
-- * '.=' — Create key-value pair for 'object'
--
-- = See Also
--
-- * "BHC.Data.JSON" for the underlying implementation
-- * "H26.Text" for text handling
-- * "H26.Bytes" for binary encoding

{-# HASKELL_EDITION 2026 #-}

module H26.JSON
  ( -- * JSON Value Type
    Value(..)
  , Object
  , Array

    -- * Encoding
  , encode
  , encodePretty
  , encodeToText
  , encodeToBytes

    -- * Decoding
  , decode
  , decode'
  , decodeStrict
  , decodeStrict'
  , eitherDecode
  , eitherDecode'

    -- * Decoding from Text/Bytes
  , decodeText
  , decodeBytes

    -- * Object Operations
  , object
  , (.=)
  , (.:)
  , (.:?)
  , (.:!)
  , (.!=)

    -- * Array Operations
  , array

    -- * FromJSON Class
  , FromJSON(..)
  , parseJSON
  , withObject
  , withArray
  , withText
  , withNumber
  , withBool

    -- * ToJSON Class
  , ToJSON(..)
  , toJSON
  , toEncoding

    -- * Parser Monad
  , Parser
  , parse
  , parseEither
  , parseMaybe
  , parseStrict

    -- * Key Type
  , Key
  , fromText
  , toText
  , fromString
  , toString

    -- * Number Handling
  , Number
  , Scientific
  , fromFloatDigits
  , toRealFloat
  , toBoundedInteger

    -- * Error Handling
  , JSONError(..)
  , JSONPath
  , JSONPathElement(..)
  , formatError
  , formatPath

    -- * Utilities
  , pairs
  , emptyObject
  , emptyArray
  , null_
  , bool_
  , number_
  , string_
  ) where

-- | JSON value representation.
data Value
  = Object !Object   -- ^ JSON object (key-value pairs)
  | Array !Array     -- ^ JSON array
  | String !Text     -- ^ JSON string
  | Number !Number   -- ^ JSON number
  | Bool !Bool       -- ^ JSON boolean
  | Null             -- ^ JSON null
  deriving (Eq, Show, Read)

-- | JSON object (map from keys to values).
type Object = Map Key Value

-- | JSON array.
type Array = Vector Value

-- | JSON object key.
newtype Key = Key Text
  deriving (Eq, Ord, Show, Read)

-- | JSON number (arbitrary precision).
data Number

-- | Arbitrary precision scientific notation.
data Scientific

-- | Encode value to JSON ByteString.
encode :: ToJSON a => a -> ByteString

-- | Encode with pretty printing.
encodePretty :: ToJSON a => a -> ByteString

-- | Encode to Text.
encodeToText :: ToJSON a => a -> Text

-- | Encode to strict ByteString.
encodeToBytes :: ToJSON a => a -> ByteString

-- | Decode JSON, returning Nothing on failure.
decode :: FromJSON a => ByteString -> Maybe a

-- | Strict decode.
decode' :: FromJSON a => ByteString -> Maybe a

-- | Decode strict ByteString.
decodeStrict :: FromJSON a => ByteString -> Maybe a

-- | Strict decode of strict ByteString.
decodeStrict' :: FromJSON a => ByteString -> Maybe a

-- | Decode with error message.
eitherDecode :: FromJSON a => ByteString -> Either String a

-- | Strict decode with error message.
eitherDecode' :: FromJSON a => ByteString -> Either String a

-- | Decode from Text.
decodeText :: FromJSON a => Text -> Either String a

-- | Decode from strict Bytes.
decodeBytes :: FromJSON a => Bytes -> Either String a

-- | Construct a JSON object from pairs.
object :: [Pair] -> Value

-- | Construct a key-value pair.
(.=) :: ToJSON a => Key -> a -> Pair

-- | Parse required field.
(.:) :: FromJSON a => Object -> Key -> Parser a

-- | Parse optional field.
(.:?) :: FromJSON a => Object -> Key -> Parser (Maybe a)

-- | Parse optional field with explicit Nothing.
(.:!) :: FromJSON a => Object -> Key -> Parser (Maybe a)

-- | Provide default for optional field.
(.!=) :: Parser (Maybe a) -> a -> Parser a

-- | Construct a JSON array.
array :: [Value] -> Value

-- | Class for types that can be decoded from JSON.
class FromJSON a where
  parseJSON :: Value -> Parser a

-- | Class for types that can be encoded to JSON.
class ToJSON a where
  toJSON :: a -> Value
  toEncoding :: a -> Encoding

-- | Parse with object accessor.
withObject :: String -> (Object -> Parser a) -> Value -> Parser a

-- | Parse with array accessor.
withArray :: String -> (Array -> Parser a) -> Value -> Parser a

-- | Parse with text accessor.
withText :: String -> (Text -> Parser a) -> Value -> Parser a

-- | Parse with number accessor.
withNumber :: String -> (Number -> Parser a) -> Value -> Parser a

-- | Parse with boolean accessor.
withBool :: String -> (Bool -> Parser a) -> Value -> Parser a

-- | JSON parser monad.
data Parser a

-- | Run parser on value.
parse :: (a -> Parser b) -> a -> Result b

-- | Run parser returning Either.
parseEither :: (a -> Parser b) -> a -> Either String b

-- | Run parser returning Maybe.
parseMaybe :: (a -> Parser b) -> a -> Maybe b

-- | Strict parsing.
parseStrict :: FromJSON a => ByteString -> Either JSONError a

-- | Convert Text to Key.
fromText :: Text -> Key

-- | Convert Key to Text.
toText :: Key -> Text

-- | Convert String to Key.
fromString :: String -> Key

-- | Convert Key to String.
toString :: Key -> String

-- | Convert floating point to Scientific.
fromFloatDigits :: RealFloat a => a -> Scientific

-- | Convert Scientific to floating point.
toRealFloat :: RealFloat a => Scientific -> a

-- | Try to convert Scientific to bounded integer.
toBoundedInteger :: (Bounded a, Integral a) => Scientific -> Maybe a

-- | JSON parsing/encoding error.
data JSONError
  = ParseError String
  | TypeError JSONPath String
  | KeyError JSONPath Key
  | IndexError JSONPath Int
  | OtherError String
  deriving (Eq, Show)

-- | Path to error location in JSON structure.
type JSONPath = [JSONPathElement]

-- | Element of JSON path.
data JSONPathElement
  = Key Key
  | Index Int
  deriving (Eq, Show)

-- | Format error message with path.
formatError :: JSONError -> String

-- | Format path for display.
formatPath :: JSONPath -> String

-- | Construct object from key-value pairs.
pairs :: [Pair] -> Value

-- | Empty JSON object.
emptyObject :: Value

-- | Empty JSON array.
emptyArray :: Value

-- | JSON null value.
null_ :: Value

-- | Construct boolean value.
bool_ :: Bool -> Value

-- | Construct number value.
number_ :: Scientific -> Value

-- | Construct string value.
string_ :: Text -> Value

-- Internal types
type Pair = (Key, Value)
data Encoding
data Result a = Error String | Success a

-- This is a specification file.
-- Actual implementation provided by the compiler.
