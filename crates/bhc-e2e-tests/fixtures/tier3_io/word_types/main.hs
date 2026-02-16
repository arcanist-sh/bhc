import Data.Word

main :: IO ()
main = do
    -- Basic Word arithmetic (all Word types use i64 at runtime)
    let w :: Word
        w = 42
    let w8 :: Word8
        w8 = 255
    let w16 :: Word16
        w16 = 1000
    let w32 :: Word32
        w32 = 100000
    let w64 :: Word64
        w64 = 999999

    -- Print values
    putStrLn (show w)
    putStrLn (show w8)
    putStrLn (show w16)
    putStrLn (show w32)
    putStrLn (show w64)

    -- Arithmetic on Word
    let x :: Word
        x = 10
    let y :: Word
        y = 3
    putStrLn (show (x + y))
    putStrLn (show (x * y))
    putStrLn (show (x - y))
