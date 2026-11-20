-- A function has no Show instance (#106).
main :: IO ()
main = print (id :: Int -> Int)
