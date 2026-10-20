-- Test: foreign import returning IO
-- Verifies: an `IO` foreign import is an action that runs when bound, under
-- both IO models (the closure model builds actions instead of running them).

foreign import ccall "abs" c_abs :: Int -> IO Int
foreign import ccall "time" c_time :: Int -> IO Int

main :: IO ()
main = do
  a <- c_abs (-42)
  print a
  t <- c_time 0
  putStrLn (if t > 1600000000 then "clock ok" else "clock bad")
