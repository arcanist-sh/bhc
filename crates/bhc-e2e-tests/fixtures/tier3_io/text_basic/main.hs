-- Test basic Data.Text operations
-- Demonstrates packed UTF-8 text vs [Char] strings

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

-- Show Int helper (simple version for small ints)
showInt :: Int -> String
showInt n = case n of
    0 -> "0"
    _ -> showHelper n []

main :: IO ()
main = do
    -- Pack a String into Text
    let t1 = Data.Text.pack "Hello"
    let t2 = Data.Text.pack "World"

    -- Get length (should be 5)
    let len = Data.Text.length t1
    putStrLn (showInt len)

    -- Append two texts
    let t3 = Data.Text.append t1 (Data.Text.pack " ")
    let t4 = Data.Text.append t3 t2

    -- Convert to uppercase
    let t5 = Data.Text.toUpper t4

    -- Unpack back to String and print
    let s = Data.Text.unpack t5
    putStrLn s

    -- Test take/drop
    let t6 = Data.Text.take 5 t4
    putStrLn (Data.Text.unpack t6)

    let t7 = Data.Text.drop 6 t4
    putStrLn (Data.Text.unpack t7)
