-- A class operation at a type given by the signature. The argument's type
-- reaches `/` through a chain of type variables, and the check looked one
-- step down that chain, so `half` compiled (#97).
half :: Int -> Int
half n = n / 2

main :: IO ()
main = print (half 4)
