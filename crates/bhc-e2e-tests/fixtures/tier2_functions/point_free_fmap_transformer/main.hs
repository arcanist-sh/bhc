-- A point-free `fmap f . g` in a function constrained only by the BASE
-- monad (`Monad m =>`) must use the Functor of the parser type `P m`, not
-- reach `fmap` through the `Monad m` dictionary (pandoc's `countChar`).
newtype P m a = P { runP :: Int -> (a, Int) }

instance Functor (P m) where
  fmap f (P g) = P (\s -> let (a, s') = g s in (f a, s'))

item :: Monad m => P m Char
item = P (\s -> (if s == 0 then 'a' else 'b', s + 1))

twice :: Monad m => P m Char -> P m [Char]
twice p = P (\s -> let (a, s1) = runP p s
                       (b, s2) = runP p s1
                   in ([a, b], s2))

lenOf :: Monad m => P m Char -> P m Int
lenOf = fmap length . twice

plus40 :: Monad m => P m Char -> P m Int
plus40 = fmap ((+ 40) . length) . twice

main :: IO ()
main = do
  print (fst (runP (lenOf (item :: P IO Char)) 0))
  print (fst (runP (plus40 (item :: P IO Char)) 0))
