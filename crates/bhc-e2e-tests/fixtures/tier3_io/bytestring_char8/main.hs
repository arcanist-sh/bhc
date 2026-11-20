-- Data.ByteString.Char8 over the strict ByteString primitives, ByteString
-- output, and show/equality of a ByteString (#109).
module Main where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC

main :: IO ()
main = do
  BC.putStrLn (BC.map succ (BC.pack "HAL"))
  BC.putStr (BC.pack "no newline")
  BC.putStrLn (BC.pack "")
  putStrLn (BC.unpack (BC.pack "round trip"))
  print (BC.length (BC.pack "four"), BC.null (BC.pack ""), BC.null BC.empty)
  BC.putStrLn (BC.append (BC.pack "ab") (BC.pack "cd"))
  BC.putStrLn (BC.reverse (BC.pack "stressed"))
  BC.putStrLn (BC.take 3 (BC.pack "abcdef"))
  BC.putStrLn (BC.drop 3 (BC.pack "abcdef"))
  BC.putStrLn (BC.cons 'x' (BC.snoc (BC.pack "yz") '!'))
  print (BC.head (BC.pack "hello"), BC.last (BC.pack "hello"))
  BC.putStrLn (BC.tail (BC.pack "hello"))
  BC.putStrLn (BC.init (BC.pack "hello"))
  print (BC.elem 'l' (BC.pack "hello"), BC.elem 'z' (BC.pack "hello"))
  print (BC.isPrefixOf (BC.pack "he") (BC.pack "hello"), BC.isSuffixOf (BC.pack "lo") (BC.pack "help"))
  print (BC.index (BC.pack "hello") 1)
  BC.putStrLn (BC.singleton 'q')
  print (BS.unpack (BC.pack "AB"))
  print (BS.unpack (BS.map (+ 1) (BS.pack [1, 2, 3])))
  BS.putStr (BS.pack [104, 105, 10])
  let greet name = BC.append (BC.pack "hi ") name
  mapM_ (BC.putStrLn . greet . BC.pack) ["a", "b"]
  print (BC.pack "HAL")
  print (BC.pack "ab" == BC.pack "ab", BC.pack "ab" == BC.pack "ac")
  print (BC.pack "ab" /= BC.pack "ab", BC.pack "ab" /= BC.pack "ac")
  putStrLn (show (BC.pack "tab\there \"quoted\""))
  let a = BC.append (BC.pack "x") (BC.pack "y")
  print (a == BC.pack "xy")
  print a
  print (BS.pack [72, 105] == BC.pack "Hi")
  print (length (filter (== BC.pack "b") (map BC.pack ["a", "b", "c", "b"])))
  print (BS.pack [200, 65])
