-- digitToInt/intToDigit cover hexadecimal digits in either case.
import Data.Char (digitToInt, intToDigit)

hexByte :: Char -> Char -> Int
hexByte a b = digitToInt a * 16 + digitToInt b

main :: IO ()
main = do
  print (map digitToInt "09afAF")
  print (hexByte 'C' '3')
  putStrLn (map intToDigit [0, 9, 10, 15])
