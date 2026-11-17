-- An equation reached when an earlier one fails at different depths. Its
-- variables are bound once per branch of the match; every branch must use
-- its own binding. `ys` resolved to the variable of one branch only, and
-- the others called a runtime stub.
--
-- The program's own `join` must also stay its own: it was rewritten as
-- Control.Monad's (`m >>= id`), which the Prelude does not even export.
data Tok = Word String | Gap
  deriving Show

newtype Toks = Toks [Tok]
  deriving Show

join :: Toks -> Toks -> Toks
join (Toks [Word a]) (Toks [Word b]) = Toks [Word (a ++ b)]
join (Toks xs) (Toks ys) = Toks (xs ++ ys)

pairUp :: Maybe Int -> [Int] -> (Int, Int)
pairUp (Just 0) [x] = (0, x)
pairUp (Just n) (y : _) = (n, y)
pairUp m zs = (maybe (-1) id m, length zs)

main :: IO ()
main = do
  print (join (Toks [Word "a"]) (Toks [Word "b"]))
  print (join (Toks [Gap]) (Toks [Word "b"]))
  print (join (Toks [Word "a", Gap]) (Toks [Gap]))
  print (join (Toks [Word "a"]) (Toks []))
  print (join (Toks [Word "a"]) (Toks [Gap, Gap]))
  print (join (Toks []) (Toks [Word "z"]))
  print (pairUp (Just 0) [7])
  print (pairUp (Just 0) [7, 8])
  print (pairUp (Just 3) [9])
  print (pairUp Nothing [1, 2, 3])
  print (pairUp (Just 4) [])
