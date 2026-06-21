main :: IO ()
main = do
  label <- getLine
  putStrLn ("Sum for " ++ label ++ ":")
  a <- readLn
  b <- readLn
  print (a + b)
  print (a - b)
