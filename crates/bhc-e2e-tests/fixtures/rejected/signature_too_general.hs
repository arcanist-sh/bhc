-- A type variable in a signature stands for every type, so a definition
-- may not choose it. Each of these compiled: the signature's variables
-- were ordinary unification variables, shared by every signature (#97).
constant :: a -> a
constant _ = True

coerce :: a -> b
coerce x = x

pick :: a -> b -> a
pick _ y = y

fromInt :: Int -> a
fromInt n = n

main :: IO ()
main = print (2 :: Int)
