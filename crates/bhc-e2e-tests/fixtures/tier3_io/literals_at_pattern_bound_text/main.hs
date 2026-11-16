{-# LANGUAGE OverloadedStrings #-}
-- String literals compared with the elements of a `[Text]` that a pattern
-- binds, where the list is used more than once. A later `null`/`length`/`any`
-- use overwrote the list's type, the earlier literals stayed `String`, and a
-- `Text` was compared as a list of characters.
import qualified Data.Text as T

type Attr = (T.Text, [T.Text], [(T.Text, T.Text)])

data Opts = Opts { lineAnchors :: Bool, numberLines :: Bool, classCount :: Int }
  deriving Show

defOpts :: Opts
defOpts = Opts False False 0

fromAttr :: Attr -> Opts
fromAttr (_, classes, _) =
  defOpts { lineAnchors = any (`elem` ["line-anchors", "lineAnchors"]) classes
          , numberLines = any (`elem` ["number", "numberLines", "number-lines"]) classes
          , classCount = length classes }

three :: (Int, [T.Text]) -> (Bool, Bool, Bool)
three (_, classes) =
  ( any (`elem` ["a1", "a2"]) classes
  , any (`elem` ["b1", "b2"]) classes
  , any (`elem` ["c1", "c2"]) classes )

viaCase :: Maybe [T.Text] -> (Bool, Bool)
viaCase m = case m of
  Just classes -> (any (== "x") classes, null classes)
  Nothing -> (False, True)

main :: IO ()
main = do
  print (fromAttr ("", ["haskell", "numberLines"], []))
  print (fromAttr ("", ["lineAnchors"], []))
  print (fromAttr ("", [], []))
  print (three (0, ["x", "b2"]))
  print (three (0, ["a1", "c2"]))
  print (viaCase (Just ["y", "x"]))
  print (viaCase (Just []))
