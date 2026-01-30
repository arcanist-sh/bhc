-- Higher-order function example
-- Demonstrates passing and invoking a closure argument multiple times

twice f x = f (f x)

main = print (twice (\x -> x * 2) 5)  -- 20
