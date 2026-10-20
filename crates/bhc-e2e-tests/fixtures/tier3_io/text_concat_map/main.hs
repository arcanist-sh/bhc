-- Data.Text.concatMap, saturated and point-free (pandoc's XML escaping).
{-# LANGUAGE OverloadedStrings #-}
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Data.Text (Text)

escape :: Char -> Text
escape '<' = "&lt;"
escape '&' = "&amp;"
escape c = T.singleton c

esc :: Text -> Text
esc = T.concatMap escape

main :: IO ()
main = do
  TIO.putStrLn (T.concatMap escape "a<b & c")
  TIO.putStrLn (esc "x<y")
