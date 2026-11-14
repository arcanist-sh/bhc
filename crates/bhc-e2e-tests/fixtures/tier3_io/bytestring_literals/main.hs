{-# LANGUAGE OverloadedStrings #-}
-- A string literal at type ByteString is a ByteString. It used to stay a
-- list of characters that the ByteString functions then read as a
-- ByteString: `BS.length ("four" :: ByteString)` was 0 and `BC.putStrLn "lit"`
-- printed an empty line.
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import qualified Data.Text as T

greeting :: BS.ByteString
greeting = "hello"

shout :: BS.ByteString -> BS.ByteString
shout b = BS.append b "!"

main :: IO ()
main = do
  BC.putStrLn "lit"
  print (BS.length ("four" :: BS.ByteString))
  BC.putStrLn (BC.append "by" (BC.pack "tes"))
  BC.putStrLn (BC.append (BC.pack "by") "tes")
  print (BS.length greeting, BC.unpack (BS.append greeting " there"))
  print (BS.isPrefixOf "he" greeting, BS.isPrefixOf "lo" greeting)
  BC.putStrLn (shout "argument")
  print (BS.unpack "AB", BS.null "", BS.null "x")
  -- Literals at the other string types are unchanged.
  print (T.length ("text" :: T.Text), length ("string" :: String))
