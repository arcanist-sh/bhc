import qualified Data.ByteString.Builder as B
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as BS

main :: IO ()
main = do
    -- Test empty builder -> toLazyByteString -> toStrict -> length = 0
    putStrLn (show (BS.length (BL.toStrict (B.toLazyByteString B.empty))))

    -- Test singleton -> 1 byte
    putStrLn (show (BS.length (BL.toStrict (B.toLazyByteString (B.singleton 65)))))

    -- Test intDec 12345 -> "12345" -> 5 bytes
    putStrLn (show (BS.length (BL.toStrict (B.toLazyByteString (B.intDec 12345)))))

    -- Test append: "42" (2 bytes) + "99" (2 bytes) = 4 bytes
    putStrLn (show (BS.length (BL.toStrict (B.toLazyByteString (B.append (B.intDec 42) (B.intDec 99))))))

    -- Test stringUtf8 "Hello" -> 5 bytes
    putStrLn (show (BS.length (BL.toStrict (B.toLazyByteString (B.stringUtf8 "Hello")))))

    -- Test stringUtf8 content length: "Builder works!" = 14 bytes
    putStrLn (show (BS.length (BL.toStrict (B.toLazyByteString (B.stringUtf8 "Builder works!")))))
