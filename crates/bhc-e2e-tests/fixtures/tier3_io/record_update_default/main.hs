-- A record update names its fields but need not say what it is updating.
-- With the base's type left to inference it stayed a variable, so the nullary
-- class method had no instance to resolve to and the update was built on the
-- wrong value — pandoc's `readMarkdown` handed its parser a state that was
-- not one, and segfaulted.
module Main where

class Blank a where
  blank :: a

data Opts = Opts { optWidth :: Int, optName :: String }
data Other = Other { othDepth :: Int }

instance Blank Opts where
  blank = Opts 0 "none"

instance Blank Other where
  blank = Other 99

main :: IO ()
main = do
  print (optWidth (blank { optWidth = 7 } :: Opts))
  putStrLn (optName (blank { optWidth = 7 } :: Opts))
  print (othDepth (blank { othDepth = 1 } :: Other))
  let o = blank { optName = "set" }
  print (optWidth (o :: Opts))
  putStrLn (optName o)
