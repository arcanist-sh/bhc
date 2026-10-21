-- `fail` at a user monad runs that monad's MonadFail instance, both as a
-- value inside a polymorphic-in-the-stream parser and applied directly.
module Main where

import qualified Control.Monad.Fail as Fail

newtype P a = P { runP :: String -> Either String (a, String) }

instance Functor P where
  fmap f (P p) = P $ \s -> case p s of
    Left e -> Left e
    Right (a, r) -> Right (f a, r)

instance Applicative P where
  pure a = P $ \s -> Right (a, s)
  P pf <*> P pa = P $ \s -> case pf s of
    Left e -> Left e
    Right (f, r) -> case pa r of
      Left e -> Left e
      Right (a, r') -> Right (f a, r')

instance Monad P where
  P p >>= k = P $ \s -> case p s of
    Left e -> Left e
    Right (a, r) -> runP (k a) r

instance Fail.MonadFail P where
  fail msg = P $ \_ -> Left msg

item :: P Char
item = P $ \s -> case s of
  (c : cs) -> Right (c, cs)
  [] -> Left "eof"

digit :: P Char
digit = do
  c <- item
  if c >= '0' && c <= '9' then return c else Prelude.fail ("not a digit: " ++ [c])

orElse :: P a -> P a -> P a
orElse (P p) (P q) = P $ \s -> either (const (q s)) Right (p s)

viaMaybe :: Maybe Int
viaMaybe = do
  x <- Just 1
  if x > (0 :: Int) then Fail.fail "positive" else Just x

failing :: P Char
failing = fail "always"

main :: IO ()
main = do
  putStrLn (either id (\(c, _) -> [c]) (runP digit "7x"))
  putStrLn (either id (\(c, _) -> [c]) (runP digit "x7"))
  putStrLn (either id (\(c, _) -> [c]) (runP (failing `orElse` item) "q"))
  putStrLn (either id (\(c, _) -> [c]) (runP failing "q"))
  print viaMaybe
