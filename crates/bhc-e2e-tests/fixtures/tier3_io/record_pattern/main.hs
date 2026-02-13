module Main where

data Person = Person { name :: String, age :: Int }

greet :: Person -> String
greet (Person { name = n, age = a }) = "Hello " ++ n ++ ", age " ++ show a

main :: IO ()
main = do
  let alice = Person { name = "Alice", age = 30 }
  let bob = Person { name = "Bob", age = 25 }
  putStrLn (greet alice)
  putStrLn (greet bob)
  putStrLn (name alice)
  putStrLn (show (age bob))
