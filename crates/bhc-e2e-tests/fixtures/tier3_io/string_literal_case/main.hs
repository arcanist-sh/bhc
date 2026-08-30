-- String-literal alternatives, both alone and mixed with constructor
-- alternatives over one `[Char]` scrutinee (pandoc's `formatFromFilePath`).
module Main where

plain :: String -> String
plain x = case x of
  ".md"  -> "markdown"
  ""     -> "empty"
  _      -> "other"

mixed :: String -> String
mixed x = case x of
  ".md"   -> "markdown"
  ".rst"  -> "rest"
  ['.',d] -> "single:" ++ [d]
  _       -> "none"

main :: IO ()
main = do
  putStrLn (plain ".md")
  putStrLn (plain (reverse "dm."))
  putStrLn (plain "")
  putStrLn (plain ".rst")
  putStrLn (mixed ".md")
  putStrLn (mixed ".rst")
  putStrLn (mixed ".3")
  putStrLn (mixed ".xyz")
