import Data.Word

main :: IO ()
main = do
    -- fromIntegral between Word and Int
    let w :: Word
        w = 42
    let n :: Int
        n = fromIntegral w
    putStrLn (show n)

    let m :: Int
        m = 100
    let w2 :: Word
        w2 = fromIntegral m
    putStrLn (show w2)

    -- Word8 conversions
    let b :: Word8
        b = 200
    let i :: Int
        i = fromIntegral b
    putStrLn (show i)
