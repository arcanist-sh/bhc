module Main where

main :: IO ()
main = do
    -- Basic character ranges
    putStrLn ['a'..'z']
    putStrLn ['A'..'Z']
    putStrLn ['0'..'9']

    -- Short ranges
    putStrLn ['a'..'e']
    putStrLn ['x'..'z']

    -- Single element range
    putStrLn ['m'..'m']

    -- Empty range (from > to)
    putStrLn ['z'..'a']

    -- Using succ and pred on Char
    putStrLn [succ 'a', succ 'b', succ 'c']
    putStrLn [pred 'z', pred 'y', pred 'x']

    -- Char range length
    print (length ['a'..'z'])
    print (length ['A'..'Z'])
    print (length ['0'..'9'])

    -- map over char range
    putStrLn (map toUpper ['a'..'f'])
