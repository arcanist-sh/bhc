{-# LANGUAGE EmptyCase #-}
data Void

absurd :: Void -> a
absurd x = case x of {}

main :: IO ()
main = putStrLn "EmptyCase works"
