-- An argument the callee never uses must not be evaluated.
boom :: Int
boom = error "BOOM"
{-# NOINLINE boom #-}

myConst :: Int -> Int -> Int
myConst a _ = a

main :: IO ()
main = do
  print (myConst 1 boom)
  putStrLn "unused arg ok"
