-- A method left to its class default, called from another module. The
-- importer took it for a plain function and passed no dictionary.
import Walk

shout :: Inline -> Inline
shout (Str s) = Str (s ++ "!")
shout x = x

main :: IO ()
main = do
  putStrLn (label (3 :: Int))
  putStrLn (label True)
  putStrLn (name (4 :: Int))
  print (walk shout (Str "a"))
  print (walk shout [Str "b", Emph [Str "c"]])
