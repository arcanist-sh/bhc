-- A definition that needs a class its signature's context does not give.
-- Both compiled: the constraint on the signature's variable was defaulted
-- (`Num a` to `Int`, `Fractional a` to `Double`) instead of required (#97).
increment :: a -> a
increment x = x + 1

half :: a -> a
half _ = 0.5

main :: IO ()
main = print (2 :: Int)
