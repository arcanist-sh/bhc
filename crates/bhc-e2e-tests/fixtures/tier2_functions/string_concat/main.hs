greet :: String -> String
greet name = "Hello, " ++ name ++ "!"

main :: IO ()
main = do
  putStrLn ("foo" ++ "bar")
  putStrLn (greet "world")
  putStrLn ("a" ++ "b" ++ "c" ++ "d")
  putStr ("no" ++ "newline")
  putStrLn ""
