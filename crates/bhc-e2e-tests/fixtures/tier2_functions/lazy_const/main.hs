-- Prelude `const` and a discarded `let` are both lazy in their unused part.
main :: IO ()
main = do
  print (const (1 :: Int) (undefined :: Int))
  let unused = error "NOPE" :: Int
  print (3 :: Int)
  putStrLn "const ok"
