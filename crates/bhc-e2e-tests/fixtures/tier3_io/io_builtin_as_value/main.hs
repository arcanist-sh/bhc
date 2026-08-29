-- An IO builtin used as a VALUE — passed to `>>=` rather than applied.
-- `putStrLn`/`putStr` reached a value-position arm that handed the cons
-- list straight to a C-string printer, so these printed nothing at all.
module Main where

acts :: IO String
acts = return "hello"

main :: IO ()
main = do
  acts >>= putStrLn
  acts >>= putStr
  putStrLn ""
  mapM_ putStrLn ["a", "b"]
  return "direct" >>= putStrLn
