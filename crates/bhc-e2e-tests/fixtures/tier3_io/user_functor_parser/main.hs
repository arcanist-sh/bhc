{-# LANGUAGE LambdaCase, BangPatterns #-}
-- A hand-written parser monad whose Functor instance maps over a Maybe, and
-- a second instance that matches on Just/Nothing (#110). The builtin `fmap`
-- at Maybe must stay Maybe's, inside the instance and elsewhere in the module.
module Main where

newtype Parser a = Parser { runParser :: String -> Maybe (a, String) }

instance Functor Parser where
  fmap f (Parser p) = Parser $ \s -> fmap (\(a, r) -> (f a, r)) (p s)

instance Applicative Parser where
  pure a = Parser $ \s -> Just (a, s)
  Parser pf <*> Parser pa = Parser $ \s -> case pf s of
    Nothing -> Nothing
    Just (f, r) -> fmap (\(a, r') -> (f a, r')) (pa r)

instance Monad Parser where
  Parser p >>= f = Parser $ \s -> case p s of
    Nothing -> Nothing
    Just (a, r) -> runParser (f a) r

newtype Q a = Q { runQ :: String -> Maybe (a, String) }

instance Functor Q where
  fmap f (Q p) = Q $ \s -> case p s of
    Nothing -> Nothing
    Just (a, r) -> Just (f a, r)

item :: Parser Char
item = Parser $ \case { [] -> Nothing; (c:cs) -> Just (c, cs) }

itemQ :: Q Char
itemQ = Q $ \s -> case s of { [] -> Nothing; (c:cs) -> Just (c, cs) }

sat :: (Char -> Bool) -> Parser Char
sat p = do { c <- item; if p c then return c else Parser (const Nothing) }

many1 :: Parser a -> Parser [a]
many1 p = do { x <- p; xs <- many0 p; return (x:xs) }

many0 :: Parser a -> Parser [a]
many0 p = Parser $ \s -> case runParser p s of
  Nothing -> Just ([], s)
  Just (a, r) -> runParser (fmap (a:) (many0 p)) r

number :: Parser Int
number = read <$> many1 (sat (`elem` "0123456789"))

expr :: Parser Int
expr = do { n <- number; rest n } where
  rest !acc = (do { _ <- sat (== '+'); m <- number; rest (acc + m) }) `orElse` pure acc
  orElse (Parser a) (Parser b) = Parser $ \s -> maybe (b s) Just (a s)

describe :: Maybe (String, String) -> String
describe Nothing = "nothing"
describe (Just (x, r)) = x ++ "|" ++ r

main :: IO ()
main = do
  print (runParser expr "1+22+333rest")
  print (runParser (many0 item) "ab")
  print (runParser number "12x")
  print (runParser (many1 (sat (== 'a'))) "bbb")
  putStrLn (describe (runParser (fmap (: []) item) "ab"))
  putStrLn (describe (runQ (fmap (: []) itemQ) "ab"))
  putStrLn (describe (fmap (\(a, r) -> ([a], r)) (Just ('a', "b"))))
  putStrLn (describe (runParser (fmap show item) "ab"))
  print (fmap (+ 1) (Just (1 :: Int)), fmap (+ 1) [1, 2 :: Int])
  print (fmap length (Just "abc"), (+ 1) <$> (Nothing :: Maybe Int))
  print (runParser ((,) <$> item <*> item) "xyz")
