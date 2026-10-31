-- writeFile/appendFile with literal and computed strings, read back (#40).
main :: IO ()
main = do
  writeFile "bhc_e2e_append.txt" "hello"
  appendFile "bhc_e2e_append.txt" " world"
  writeFile "bhc_e2e_computed.txt" (reverse "hello" ++ "!")
  s1 <- readFile "bhc_e2e_append.txt"
  s2 <- readFile "bhc_e2e_computed.txt"
  putStrLn s1
  putStrLn s2
  putStrLn ("<" ++ reverse "ab" ++ ">")
  print (length (reverse "xy" ++ "z"))
