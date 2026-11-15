-- The two arguments of `compare` have one type. This compiled: `compare`
-- was typed as a fresh variable, so nothing made them agree (#97).
main :: IO ()
main = print (compare (3 :: Int) "x")
