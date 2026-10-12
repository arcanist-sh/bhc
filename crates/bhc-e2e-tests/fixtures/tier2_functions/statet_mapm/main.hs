-- `mapM` over a builtin transformer must run each action in order (codegen's
-- builtin `mapM` is a plain `map`).
module Main where
import Control.Monad.State (StateT, evalStateT, modify, get)

stepM :: Int -> StateT Int IO Int
stepM x = do
  modify (+ x)
  s <- get
  return (s * 10)

main :: IO ()
main = do
  a <- evalStateT (stepM 1 >>= \u -> stepM 2 >>= \v -> return (u + v)) 0
  print a
  b <- evalStateT (mapM stepM [1, 2, 3]) 0
  print (sum b)
  c <- evalStateT (mapM stepM [5]) 1
  print (sum c)
