{-# LANGUAGE DefaultSignatures #-}

class HasName a where
  name :: a -> String
  default name :: Show a => a -> String
  name _ = "unnamed"

main :: IO ()
main = putStrLn "DefaultSignatures works"
