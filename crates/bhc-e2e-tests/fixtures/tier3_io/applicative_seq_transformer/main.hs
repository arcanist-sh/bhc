-- `a <* b` under a transformer: both actions run, the state threads through
-- both, and the FIRST one's value is the result. Only the IO layer was
-- lowered, so pandoc's `setIntraword True *> p <* setIntraword False` aborted
-- with "unknown builtin: <*".
module Main where
import Control.Monad.State

tick :: StateT Int IO ()
tick = modify (+ 1)

keepFirst :: StateT Int IO Int
keepFirst = (tick >> return (7 :: Int)) <* tick

seqBoth :: StateT Int IO Int
seqBoth = tick *> return (9 :: Int)

main :: IO ()
main = do
  (v, s) <- runStateT keepFirst 0
  print v
  print s
  (v2, s2) <- runStateT seqBoth 0
  print v2
  print s2
