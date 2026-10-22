-- An explicitly recursive parser: `go n = p >> go n`. Building `go 1`'s
-- parser value must not evaluate `go 1` again (an eager argument recursed
-- forever), and the literal-pattern equations' `n` must be the argument.
newtype P a = P { runP :: String -> Maybe (a, String) }

instance Functor P where
  fmap f (P p) = P (\s -> case p s of
                            Nothing -> Nothing
                            Just (a, rest) -> Just (f a, rest))

instance Applicative P where
  pure x = P (\s -> Just (x, s))
  P pf <*> P pa = P (\s -> case pf s of
                             Nothing -> Nothing
                             Just (f, rest) -> case pa rest of
                               Nothing -> Nothing
                               Just (a, rest') -> Just (f a, rest'))

instance Monad P where
  P p >>= k = P (\s -> case p s of
                         Nothing -> Nothing
                         Just (a, rest) -> runP (k a) rest)

orElse :: P a -> P a -> P a
orElse (P p) (P q) = P (\s -> case p s of
                                Nothing -> q s
                                r -> r)

sym :: Char -> P Char
sym c = P (\s -> case s of
                   (x:rest) | x == c -> Just (x, rest)
                   _ -> Nothing)

item :: P Char
item = P (\s -> case s of
                  (x:rest) | x /= ']' && x /= '[' -> Just (x, rest)
                  _ -> Nothing)

-- Count bracket nesting; `go 0` closes.
brackets :: Int -> P Int
brackets 0 = return 0
brackets n = (sym ']' >> fmap (+ 1) (brackets (n - 1)))
             `orElse` (sym '[' >> brackets (n + 1))
             `orElse` (item >> brackets n)

balanced :: P Int
balanced = sym '[' >> go 1
  where
    go :: Int -> P Int
    go 0 = return 100
    go n = (sym ']' >> go (n - 1)) `orElse` (sym '[' >> go (n + 1)) `orElse` (item >> go n)

result :: Maybe (Int, String) -> String
result Nothing = "none"
result (Just (v, rest)) = show v ++ " rest=" ++ rest

main :: IO ()
main = do
  putStrLn (result (runP (sym '[' >> brackets 1) "[ab]cd"))
  putStrLn (result (runP (sym '[' >> brackets 1) "[a[b]c]!"))
  putStrLn (result (runP balanced "[x[y]z]tail"))
  putStrLn (result (runP balanced "[unclosed"))
