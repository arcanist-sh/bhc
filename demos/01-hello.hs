-- 30-second demo: real Haskell compiles to a native binary via LLVM.
--
--   bhc demos/01-hello.hs -o /tmp/hello
--   /tmp/hello
--
-- Output:
--   Hello from BHC at Zurihac 2026!
--   1 + 2 + 3 + ... + 100 = 5050
module Main where

main :: IO ()
main = do
  putStrLn "Hello from BHC at Zurihac 2026!"
  let n = sum [1..100]
  putStrLn ("1 + 2 + 3 + ... + 100 = " ++ show n)
