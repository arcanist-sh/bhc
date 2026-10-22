-- A parser whose definition refers to itself under `>>`: building it must
-- not run it (pandoc's quotedTitle).
newtype P a = P { runP :: String -> Maybe (a, String) }

instance Functor P where
  fmap f (P p) = P $ \s -> case p s of
    Nothing -> Nothing
    Just (a, r) -> Just (f a, r)

instance Applicative P where
  pure a = P $ \s -> Just (a, s)
  P pf <*> P pa = P $ \s -> case pf s of
    Nothing -> Nothing
    Just (f, r) -> case pa r of
      Nothing -> Nothing
      Just (a, r') -> Just (f a, r')

instance Monad P where
  P p >>= k = P $ \s -> case p s of
    Nothing -> Nothing
    Just (a, r) -> runP (k a) r

class Monad m => Choice m where
  orElse :: m a -> m a -> m a

instance Choice P where
  orElse (P p) (P q) = P $ \s -> case p s of
    Nothing -> q s
    ok -> ok

class Monad m => CharParser m where
  sat :: (Char -> Bool) -> m Char

instance CharParser P where
  sat f = P $ \s -> case s of
    (c : cs) | f c -> Just (c, cs)
    _ -> Nothing

char :: CharParser m => Char -> m Char
char c = sat (== c)

manyTill :: Choice m => m a -> m b -> m [a]
manyTill p end = go
  where
    go = orElse ([] <$ end) ((:) <$> p <*> go)

nestedPoly :: (CharParser m, Choice m) => m String
nestedPoly = char '(' >>
  (let inner = (\x -> "(" ++ x ++ ")") <$> nestedPoly
   in concat <$> manyTill (orElse inner ((: []) <$> sat (`notElem` "()"))) (char ')'))

nestedMono :: P String
nestedMono = char '(' >>
  (let inner = (\x -> "(" ++ x ++ ")") <$> nestedMono
   in concat <$> manyTill (orElse inner ((: []) <$> sat (`notElem` "()"))) (char ')'))

main :: IO ()
main = do
  putStrLn (maybe "fail" fst (runP nestedPoly "(ab(cd)ef)"))
  putStrLn (maybe "fail" fst (runP nestedMono "(ab(cd)ef)"))
