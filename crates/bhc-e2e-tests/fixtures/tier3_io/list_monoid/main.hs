module Main where

-- `mempty` has no argument to dispatch on and no dictionary in scope at a
-- top-level use. Result-type dispatch was gated to USER classes, so `mempty`
-- at a builtin class never reached instance resolution: it stayed a bare
-- `mempty` in Core and `length` walked it as garbage, answering 5.
main :: IO ()
main = do
  print (length (mempty :: [Int]))
  print (null (mempty :: [Int]))
  print (mempty <> [1, 2 :: Int])
  print ([1 :: Int] <> mempty)
  print (mconcat [[1], [2, 3 :: Int]])
  print (mappend [1 :: Int] [2])
