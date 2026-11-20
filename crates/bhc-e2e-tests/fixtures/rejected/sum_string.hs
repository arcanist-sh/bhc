-- A Char is not a number: `sum` over a String has no Num instance (#97).
main :: IO ()
main = print (sum "abc")
