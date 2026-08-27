module Main where

import Control.Monad.Writer

-- `Writer w` is the mtl alias for `WriterT w Identity`; runWriter and
-- execWriter used to be unimplemented stubs.
--
-- A String log, matching the writer_t fixture: a list log such as [Int]
-- still prints its accumulator wrongly, but it does so for WriterT too —
-- that is the unimplemented builtin list Semigroup, not the alias.
go :: Writer String Int
go = do
  tell "hello "
  tell "world"
  return 7

main :: IO ()
main = do
  let (a, w) = runWriter go
  print a
  putStrLn w
  putStrLn (execWriter go)
