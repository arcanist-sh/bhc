{-# LANGUAGE OverloadedStrings #-}
-- Test: a local function passed as a value keeps its value-class variables
-- monomorphic. `toDoc` compares with a string literal and is mapped over
-- Text; generalizing it let the literal default the variable to String while
-- the use was at Text (pandoc's Writers.Blaze `toChunks`).

import qualified Data.Text as T
import Data.Text (Text)

class Render a where
  render :: a -> String

instance Render T.Text where
  render t = "T:" ++ T.unpack t

instance Render [c] where
  render _ = "list"

data Chunk = Space | Word String

chunks :: Text -> [Chunk]
chunks t0 = map toDoc (T.groupBy sameStatus t0)
  where
    toDoc t
      | t == " " = Space
      | otherwise = Word (render t)
    sameStatus c d = (c == ' ') == (d == ' ')

shw :: Chunk -> String
shw Space = "_"
shw (Word w) = w

main :: IO ()
main = putStrLn (concatMap shw (chunks "hello world"))
