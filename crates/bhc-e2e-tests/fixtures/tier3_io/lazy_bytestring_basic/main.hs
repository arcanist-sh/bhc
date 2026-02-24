import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as BS

-- Simple showInt helper
digitChar :: Int -> Char
digitChar d = case d of
    0 -> '0'
    1 -> '1'
    2 -> '2'
    3 -> '3'
    4 -> '4'
    5 -> '5'
    6 -> '6'
    7 -> '7'
    8 -> '8'
    9 -> '9'
    _ -> '?'

showInt :: Int -> String
showInt n
    | n < 0     = '-' : showInt (negate n)
    | n < 10    = [digitChar n]
    | otherwise = showInt (n `div` 10) ++ [digitChar (n `mod` 10)]

main :: IO ()
main = do
    -- fromStrict / toStrict roundtrip
    let bs = BS.pack [72, 101, 108, 108, 111]
    let lazy1 = BL.fromStrict bs
    let back = BL.toStrict lazy1
    putStrLn (showInt (BS.length back))

    -- length of empty
    putStrLn (showInt (BL.length BL.empty))

    -- length of non-empty
    putStrLn (showInt (BL.length lazy1))

    -- head
    putStrLn (showInt (BL.head lazy1))

    -- append
    let lazy2 = BL.append lazy1 (BL.fromStrict (BS.singleton 33))
    putStrLn (showInt (BL.length lazy2))
