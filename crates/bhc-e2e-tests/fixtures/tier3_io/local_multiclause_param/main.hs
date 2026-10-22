-- Test: a local multi-clause function's constructor patterns fix its
-- parameter type.
-- Verifies: `level (Header n) = n; level _ = 99` in a where clause is
-- `Block -> Int`, so a class method chosen by that type (`summarize`) finds
-- its instance. The generated parameter used to get a polymorphic type and
-- the instance was never resolved.

data Block = Header Int | Para String

class Summary a where
  summarize :: (a -> Int) -> [a] -> Int

instance Summary Block where
  summarize f = minimum . map f

minLevel :: [Block] -> Int
minLevel bs = summarize level bs
  where
    level (Header n) = n
    level _ = 99

firstWord :: [Block] -> String
firstWord bs = concatMap pick bs
  where
    pick (Para (c : _)) = [c]
    pick _ = ""

main :: IO ()
main = do
  print (minLevel [Para "x", Header 3, Header 2])
  putStrLn (firstWord [Para "hello", Header 1, Para "world"])
