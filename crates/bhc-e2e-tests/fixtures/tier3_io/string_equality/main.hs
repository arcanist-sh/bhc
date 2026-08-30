-- A string literal is a `[Char]` cons list, so `==` on two of them compared
-- two heap POINTERS and was False for every pair of equal strings.
module Main where

eqs :: String -> String -> Bool
eqs a b = a == b

main :: IO ()
main = do
  print ("abc" == "abc")
  print ("abc" == "abd")
  print (reverse "cba" == "abc")
  print (eqs "abc" ("ab" ++ "c"))
  print ("abc" /= "abd")
  print (['a','b'] == "ab")
