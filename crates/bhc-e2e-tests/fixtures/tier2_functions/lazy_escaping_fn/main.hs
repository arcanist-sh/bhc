-- `myConst` has an unused (lazy) second parameter AND is used as a value.
-- If escape analysis is wrong, the indirect call passes a value where the
-- callee expects a thunk, and forcing it corrupts.
myConst :: Int -> Int -> Int
myConst a _ = a

applyTwice :: (Int -> Int -> Int) -> Int
applyTwice f = f 1 2 + f 3 4

main :: IO ()
main = do
  print (myConst 10 20)
  print (applyTwice myConst)
  print (map (\g -> g 7 8) [myConst])
