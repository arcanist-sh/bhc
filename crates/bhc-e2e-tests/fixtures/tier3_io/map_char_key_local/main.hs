-- A `Map Char` lookup on a typed `Char` binder of a local multi-clause
-- function (pandoc's Roff writer escape table).
module Main where

import qualified Data.Map as M

table :: M.Map Char String
table = M.fromList [('-', "\\-"), ('.', "\\&.")]

escape :: String -> String
escape = concatMap esc
  where
    esc '\\' = "\\e"
    esc c = maybe [c] id (M.lookup c table)

render :: Bool -> String -> String
render = go
  where
    go True s = escape s
    go False s = s

main :: IO ()
main = do
  putStrLn (escape "a-b.c\\")
  putStrLn (render True "x-y")
  putStrLn (render False "x-y")
