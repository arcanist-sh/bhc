-- A partial application records how many arguments it still WANTS, and `(.)`
-- and `($)` apply through the closure machinery rather than a fixed
-- one-argument call. parsec's `parsecMap` passes `cok . f` — a 2-of-3 partial
-- `(.)` — as a continuation which is then invoked with all three of its
-- arguments; without this, `three 1 . id` called `three 1` with ONE argument
-- and the lifted body read the other two from whatever the registers held.
module Main where

three :: Int -> Int -> Int -> Int
three a b c = a * 100 + b * 10 + c

main :: IO ()
main = do
  -- A user partial application, over-applied.
  let p = three 7
  print (p 8 9)
  -- A composition, over-applied inline...
  print ((three 1 . id) (2 :: Int) 3)
  -- ...and the same composition bound first, so the application goes through
  -- the closure path rather than the builtin's own over-application split.
  let q = three 1 . (+ (1 :: Int))
  print (q 2 3)
  let r = three 9 . id
  print (r 2 4)
  -- `($)` has the same hazard.
  print ((three 5 $ 6) 7)
