-- `TL.Text` in another module's signature is lazy Text. It was lowered as
-- strict Text, so `toStrict` of the result failed to type-check (pandoc's
-- ANSI writer: `toStrict $ D.renderANSI …`).
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Lazy as TL
import Data.Text.Lazy (toStrict)
import LazyLib

shout :: TL.Text -> TL.Text
shout t = TL.pack (map succ (TL.unpack t))

main :: IO ()
main = do
  TIO.putStrLn (toStrict (lazyOf 42))
  TIO.putStrLn (toStrict (lazyShown Nothing (7 :: Int)))
  print (T.length (toStrict (shout (lazyOf 1234))))
  TIO.putStrLn (toStrict (shout (TL.pack "HAL")))
  putStrLn (roundTrip "round trip")
