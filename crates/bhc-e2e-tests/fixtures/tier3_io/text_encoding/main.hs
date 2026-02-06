-- Test Data.Text.Encoding: encodeUtf8 / decodeUtf8

-- Simple Int to Char conversion for single digits
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

-- Show helper for building string
showHelper :: Int -> String -> String
showHelper 0 acc = acc
showHelper x acc =
    let d = x `mod` 10
        c = digitChar d
    in showHelper (x `div` 10) (c : acc)

-- Show Int helper
showInt :: Int -> String
showInt n = case n of
    0 -> "0"
    _ -> showHelper n []

main :: IO ()
main = do
    -- Pack a string into Text
    let t = Data.Text.pack "Hello"
    -- Encode Text to ByteString
    let bs = Data.Text.Encoding.encodeUtf8 t
    -- ByteString length should be 5 (ASCII = 1 byte per char)
    putStrLn (showInt (Data.ByteString.length bs))
    -- Decode ByteString back to Text
    let t2 = Data.Text.Encoding.decodeUtf8 bs
    -- Unpack and print
    putStrLn (Data.Text.unpack t2)
