{-# LANGUAGE OverloadedStrings #-}
-- A Text literal pattern nested under a list cons, a tuple, or a
-- constructor whose field is a type alias must compare as Text.
import Data.Text (Text)

type Attr = (Text, [Text], [(Text, Text)])

data Blk = Div Attr [Blk] | Hdr Int

headIsSection :: [Text] -> String
headIsSection ("section" : _) = "yes"
headIsSection _ = "no"

secondIsSection :: (Text, [Text]) -> String
secondIsSection (_, "section" : _) = "yes"
secondIsSection _ = "no"

kind :: Blk -> String
kind (Div (_, "section" : _, _) (Hdr _ : _)) = "section"
kind (Div _ _) = "div"
kind _ = "other"

main :: IO ()
main = do
  putStrLn (headIsSection ["section", "x"])
  putStrLn (headIsSection ["x"])
  putStrLn (secondIsSection ("", ["section"]))
  putStrLn (kind (Div ("", ["section"], []) [Hdr 1]))
  putStrLn (kind (Div ("", ["x"], []) [Hdr 1]))
  putStrLn (kind (Hdr 2))
