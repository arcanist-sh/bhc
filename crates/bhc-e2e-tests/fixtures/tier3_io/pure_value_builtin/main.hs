-- `pure` passed as a value at a list or Maybe type builds a singleton /
-- Just (pandoc's uri parser: `fmap pure (satisfy p)` then `T.pack`).
module Main where

newtype P a = P { runP :: String -> Maybe (a, String) }

instance Functor P where
  fmap f (P p) = P $ \s -> case p s of
    Nothing -> Nothing
    Just (a, r) -> Just (f a, r)

item :: P Char
item = P $ \s -> case s of
  (c : cs) -> Just (c, cs)
  [] -> Nothing

single :: P String
single = fmap pure item

singleM :: Maybe Char -> Maybe String
singleM = fmap pure

main :: IO ()
main = do
  putStrLn (maybe "none" fst (runP single "xy"))
  print (length (maybe "" fst (runP single "xy")))
  putStrLn (maybe "none" id (singleM (Just 'q')))
  putStrLn (concatMap (++ ",") (map pure "ab" :: [String]))
