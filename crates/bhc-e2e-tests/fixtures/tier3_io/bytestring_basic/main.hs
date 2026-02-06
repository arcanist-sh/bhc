-- Test basic Data.ByteString operations

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
    -- Pack a list of Word8 values into a ByteString
    let bs1 = Data.ByteString.pack [72, 101, 108, 108, 111]
    let len = Data.ByteString.length bs1
    putStrLn (showInt len)

    -- Append a byte
    let bs2 = Data.ByteString.append bs1 (Data.ByteString.singleton 33)
    putStrLn (showInt (Data.ByteString.length bs2))

    -- Head byte
    let h = Data.ByteString.head bs1
    putStrLn (showInt h)

    -- Take first 3 bytes
    let bs3 = Data.ByteString.take 3 bs1
    putStrLn (showInt (Data.ByteString.length bs3))

    -- Reverse and get head
    let bs4 = Data.ByteString.reverse bs1
    putStrLn (showInt (Data.ByteString.head bs4))
