-- |
-- Module      : BHC.Data.JSON
-- Description : JSON parsing and serialization
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- JSON encoding and decoding.

module BHC.Data.JSON (
    -- * JSON type
    JSON(..),
    
    -- * Parsing
    decode,
    decodeStrict,
    eitherDecode,
    
    -- * Serialization
    encode,
    encodePretty,
    
    -- * Accessors
    (.:), (.:?), (.!=),
    
    -- * Construction
    object, (.=),
    array,
    
    -- * Type classes
    ToJSON(..),
    FromJSON(..),
    
    -- * Errors
    JSONError(..),
) where

import BHC.Prelude hiding (null)
import qualified BHC.Prelude as P
import qualified BHC.Data.Map as Map

-- | A JSON value.
data JSON
    = Null
    | Bool !Bool
    | Number !Double
    | String !String
    | Array ![JSON]
    | Object !(Map.Map String JSON)
    deriving (Eq, Show, Read)

-- | JSON parsing errors.
data JSONError
    = ParseError String
    | KeyNotFound String
    | TypeMismatch String String
    deriving (Eq, Show, Read)

-- | Decode JSON from a string.
decode :: FromJSON a => String -> Maybe a
decode s = case eitherDecode s of
    Right x -> Just x
    Left _  -> Nothing

-- | Strict decode (errors on failure).
decodeStrict :: FromJSON a => String -> a
decodeStrict s = case eitherDecode s of
    Right x -> x
    Left e  -> error (show e)

-- | Decode with error information.
eitherDecode :: FromJSON a => String -> Either JSONError a
eitherDecode s = case parseJSON s of
    Left e  -> Left (ParseError e)
    Right j -> fromJSON j

-- | Encode to JSON string.
encode :: ToJSON a => a -> String
encode = encodeJSON . toJSON

-- | Encode with pretty printing.
encodePretty :: ToJSON a => a -> String
encodePretty = encodePrettyJSON . toJSON

-- | Parse raw JSON.
parseJSON :: String -> Either String JSON
parseJSON s = case runParser jsonValue (dropWhile isSpace s) of
    Just (j, "") -> Right j
    Just (_, r)  -> Left ("unexpected: " ++ take 20 r)
    Nothing      -> Left "parse error"

-- | Encode JSON value to string.
encodeJSON :: JSON -> String
encodeJSON Null = "null"
encodeJSON (Bool True) = "true"
encodeJSON (Bool False) = "false"
encodeJSON (Number n) = show n
encodeJSON (String s) = "\"" ++ escapeString s ++ "\""
encodeJSON (Array xs) = "[" ++ intercalate "," (P.map encodeJSON xs) ++ "]"
encodeJSON (Object m) = "{" ++ intercalate "," (P.map pair (Map.toList m)) ++ "}"
  where pair (k, v) = "\"" ++ escapeString k ++ "\":" ++ encodeJSON v

-- | Encode with indentation.
encodePrettyJSON :: JSON -> String
encodePrettyJSON = go 0
  where
    go _ Null = "null"
    go _ (Bool True) = "true"
    go _ (Bool False) = "false"
    go _ (Number n) = show n
    go _ (String s) = "\"" ++ escapeString s ++ "\""
    go indent (Array []) = "[]"
    go indent (Array xs) =
        "[\n" ++ intercalate ",\n" (P.map (\x -> spaces (indent + 2) ++ go (indent + 2) x) xs) ++ "\n" ++ spaces indent ++ "]"
    go indent (Object m) | Map.null m = "{}"
    go indent (Object m) =
        "{\n" ++ intercalate ",\n" (P.map pair (Map.toList m)) ++ "\n" ++ spaces indent ++ "}"
      where pair (k, v) = spaces (indent + 2) ++ "\"" ++ escapeString k ++ "\": " ++ go (indent + 2) v
    spaces n = P.replicate n ' '

escapeString :: String -> String
escapeString = concatMap escape
  where
    escape '"'  = "\\\""
    escape '\\' = "\\\\"
    escape '\n' = "\\n"
    escape '\r' = "\\r"
    escape '\t' = "\\t"
    escape c    = [c]

-- Accessors
(.:) :: FromJSON a => Map.Map String JSON -> String -> Either JSONError a
m .: k = case Map.lookup k m of
    Nothing -> Left (KeyNotFound k)
    Just v  -> fromJSON v

(.:?) :: FromJSON a => Map.Map String JSON -> String -> Either JSONError (Maybe a)
m .:? k = case Map.lookup k m of
    Nothing -> Right Nothing
    Just v  -> fmap Just (fromJSON v)

(.!=) :: Either JSONError (Maybe a) -> a -> Either JSONError a
ea .!= def = fmap (maybe def id) ea

-- Construction
object :: [(String, JSON)] -> JSON
object = Object . Map.fromList

(.=) :: ToJSON a => String -> a -> (String, JSON)
k .= v = (k, toJSON v)

array :: [JSON] -> JSON
array = Array

-- Type classes
class ToJSON a where
    toJSON :: a -> JSON

class FromJSON a where
    fromJSON :: JSON -> Either JSONError a

-- Instances
instance ToJSON JSON where
    toJSON = id

instance FromJSON JSON where
    fromJSON = Right

instance ToJSON Bool where
    toJSON = Bool

instance FromJSON Bool where
    fromJSON (Bool b) = Right b
    fromJSON j = Left (TypeMismatch "Bool" (typeOf j))

instance ToJSON Int where
    toJSON = Number . fromIntegral

instance FromJSON Int where
    fromJSON (Number n) = Right (round n)
    fromJSON j = Left (TypeMismatch "Int" (typeOf j))

instance ToJSON Integer where
    toJSON = Number . fromIntegral

instance FromJSON Integer where
    fromJSON (Number n) = Right (round n)
    fromJSON j = Left (TypeMismatch "Integer" (typeOf j))

instance ToJSON Double where
    toJSON = Number

instance FromJSON Double where
    fromJSON (Number n) = Right n
    fromJSON j = Left (TypeMismatch "Double" (typeOf j))

instance ToJSON String where
    toJSON = String

instance FromJSON String where
    fromJSON (String s) = Right s
    fromJSON j = Left (TypeMismatch "String" (typeOf j))

instance ToJSON a => ToJSON [a] where
    toJSON = Array . P.map toJSON

instance FromJSON a => FromJSON [a] where
    fromJSON (Array xs) = traverse fromJSON xs
    fromJSON j = Left (TypeMismatch "Array" (typeOf j))

instance ToJSON a => ToJSON (Maybe a) where
    toJSON Nothing = Null
    toJSON (Just x) = toJSON x

instance FromJSON a => FromJSON (Maybe a) where
    fromJSON Null = Right Nothing
    fromJSON j = fmap Just (fromJSON j)

typeOf :: JSON -> String
typeOf Null = "null"
typeOf (Bool _) = "bool"
typeOf (Number _) = "number"
typeOf (String _) = "string"
typeOf (Array _) = "array"
typeOf (Object _) = "object"

-- Simple parser
type Parser a = String -> Maybe (a, String)

runParser :: Parser a -> String -> Maybe (a, String)
runParser = id

jsonValue :: Parser JSON
jsonValue s = case dropWhile isSpace s of
    'n':'u':'l':'l':rest -> Just (Null, rest)
    't':'r':'u':'e':rest -> Just (Bool True, rest)
    'f':'a':'l':'s':'e':rest -> Just (Bool False, rest)
    '"':rest -> parseString rest
    '[':rest -> parseArray rest
    '{':rest -> parseObject rest
    s' -> parseNumber s'

parseString :: Parser JSON
parseString = fmap (first String) . go ""
  where
    go acc ('"':rest) = Just (P.reverse acc, rest)
    go acc ('\\':c:rest) = go (unescape c : acc) rest
    go acc (c:rest) = go (c:acc) rest
    go _ "" = Nothing
    unescape 'n' = '\n'
    unescape 'r' = '\r'
    unescape 't' = '\t'
    unescape c = c

parseNumber :: Parser JSON
parseNumber s =
    let (num, rest) = P.span isNumChar s
    in if P.null num then Nothing else Just (Number (read num), rest)
  where isNumChar c = c `P.elem` "-0123456789.eE+"

parseArray :: Parser JSON
parseArray s = case dropWhile isSpace s of
    ']':rest -> Just (Array [], rest)
    _ -> do
        (first, rest1) <- jsonValue s
        go [first] rest1
  where
    go acc s' = case dropWhile isSpace s' of
        ']':rest -> Just (Array (P.reverse acc), rest)
        ',':rest -> do
            (val, rest') <- jsonValue (dropWhile isSpace rest)
            go (val:acc) rest'
        _ -> Nothing

parseObject :: Parser JSON
parseObject s = case dropWhile isSpace s of
    '}':rest -> Just (Object Map.empty, rest)
    '"':_ -> do
        (k, rest1) <- parseKey s
        rest2 <- expect ':' rest1
        (v, rest3) <- jsonValue (dropWhile isSpace rest2)
        go [(k, v)] rest3
    _ -> Nothing
  where
    parseKey ('"':rest) = go "" rest
      where go acc ('"':r) = Just (P.reverse acc, dropWhile isSpace r)
            go acc (c:r) = go (c:acc) r
            go _ "" = Nothing
    parseKey _ = Nothing
    
    expect c s' = case dropWhile isSpace s' of
        (c':rest) | c == c' -> Just rest
        _ -> Nothing
    
    go acc s' = case dropWhile isSpace s' of
        '}':rest -> Just (Object (Map.fromList acc), rest)
        ',':rest -> do
            (k, rest1) <- parseKey (dropWhile isSpace rest)
            rest2 <- expect ':' rest1
            (v, rest3) <- jsonValue (dropWhile isSpace rest2)
            go ((k,v):acc) rest3
        _ -> Nothing

isSpace :: Char -> Bool
isSpace c = c `P.elem` " \t\n\r"

intercalate :: [a] -> [[a]] -> [a]
intercalate _ [] = []
intercalate _ [x] = x
intercalate sep (x:xs) = x ++ sep ++ intercalate sep xs

first :: (a -> b) -> (a, c) -> (b, c)
first f (a, c) = (f a, c)
