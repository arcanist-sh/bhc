-- Pattern-matching Fibonacci example
-- Demonstrates multi-clause function definitions with literal patterns

fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)

main = print (fib 10)  -- 55
