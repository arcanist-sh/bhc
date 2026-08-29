-- `*>` and `<*` at IO. Both used to abort with `stub: *> not implemented`:
-- the Applicative class lists them, but nothing lowered them.
module Main where

main :: IO ()
main = do
  putStrLn "a" *> putStrLn "b"
  putStrLn "c" <* putStrLn "d"
  keep <- return (1 :: Int) <* return (2 :: Int)
  print keep
  drop1 <- return (1 :: Int) *> return (2 :: Int)
  print drop1
  putStrLn "x" *> putStrLn "y" *> putStrLn "z"
