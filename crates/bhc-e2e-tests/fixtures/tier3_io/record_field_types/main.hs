module Main where

data Person = Person { name :: String, age :: Int }

data Box a = Box { value :: a }

main :: IO ()
main = do
  -- Record construction + accessor usage (verifies types flow correctly)
  let p = Person { name = "Alice", age = 30 }
  putStrLn (name p)
  putStrLn (show (age p))

  -- Record update with correct types
  let p2 = p { age = 31 }
  putStrLn (name p2)
  putStrLn (show (age p2))

  -- Polymorphic record field types
  let b1 = Box { value = 42 :: Int }
  putStrLn (show (value b1))
  let b2 = Box { value = "hello" }
  putStrLn (value b2)

  -- Polymorphic record update
  let b3 = b1 { value = 99 }
  putStrLn (show (value b3))
