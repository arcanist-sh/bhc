-- | Functions whose signatures name lazy Text through a qualified import.
-- The encoding module shares the `TL` alias, as in pandoc's UTF8 module.
module LazyLib (lazyOf, lazyShown, roundTrip) where

import qualified Data.ByteString.Lazy.Char8 as BL
import qualified Data.Text.Lazy as TL
import qualified Data.Text.Lazy.Encoding as TL

lazyOf :: Int -> TL.Text
lazyOf n = TL.pack (show n)

lazyShown :: Show a => Maybe Int -> a -> TL.Text
lazyShown _ x = TL.pack (show x)

toBytes :: TL.Text -> BL.ByteString
toBytes = TL.encodeUtf8

fromBytes :: BL.ByteString -> TL.Text
fromBytes = TL.decodeUtf8

roundTrip :: String -> String
roundTrip = TL.unpack . fromBytes . toBytes . TL.pack
