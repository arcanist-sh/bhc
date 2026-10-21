-- Text literal patterns in a local multi-clause function's later arguments
-- match (pandoc's strToHtml helper), around a guarded clause.
{-# LANGUAGE OverloadedStrings #-}
module Main where
import qualified Data.Text as T

render :: [T.Text] -> String
render = foldl go ""
  where
    go h "'" = h ++ "<apos>"
    go h "\"" = h ++ "<quot>"
    go h txt | T.length txt == 1 && T.all (== '~') txt = h ++ "<tilde>"
    go h txt = h ++ T.unpack txt

main :: IO ()
main = putStrLn (render ["a", "\"", "b", "~", "'", "cd"])
