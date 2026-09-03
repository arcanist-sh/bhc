-- A constructor field that is never projected must not be evaluated.
boom :: Int
boom = error "BOOM"
{-# NOINLINE boom #-}

main :: IO ()
main = do
  print (fst (2 :: Int, boom))
  print (length [boom, boom])
  putStrLn "con field ok"
