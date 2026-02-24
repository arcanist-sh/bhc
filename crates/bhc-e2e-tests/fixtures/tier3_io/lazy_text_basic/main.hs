import qualified Data.Text.Lazy as TL
import qualified Data.Text as T

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
    let txt = T.pack "Hello, Lazy Text!"
    let lazy1 = TL.fromStrict txt
    let back = TL.toStrict lazy1
    putStrLn (T.unpack back)

    -- length of empty
    putStrLn (showInt (TL.length TL.empty))

    -- length of non-empty
    putStrLn (showInt (TL.length lazy1))

    -- pack / unpack roundtrip
    let lazy2 = TL.pack "Packed lazy"
    putStrLn (TL.unpack lazy2)

    -- append via toStrict
    let lazy3 = TL.append (TL.pack "Hello") (TL.pack " World")
    putStrLn (T.unpack (TL.toStrict lazy3))
