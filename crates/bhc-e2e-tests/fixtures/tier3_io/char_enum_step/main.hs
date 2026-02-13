module Main where

main :: IO ()
main = do
    -- Stepped char ranges (enumFromThenTo)
    putStrLn ['a','c'..'z']
    putStrLn ['a','d'..'z']
    putStrLn ['A','C'..'Z']
    putStrLn ['0','2'..'9']

    -- Reverse stepped ranges
    putStrLn ['z','y'..'t']
    putStrLn ['z','x'..'a']

    -- Lengths
    print (length ['a','c'..'z'])
    print (length ['z','x'..'a'])

    -- Using with take on infinite char enum (enumFromThen)
    putStrLn (take 5 ['a','c'..])
    putStrLn (take 10 ['A'..])

    -- succ/pred on Int (verify still works)
    print (succ 5)
    print (pred 10)
