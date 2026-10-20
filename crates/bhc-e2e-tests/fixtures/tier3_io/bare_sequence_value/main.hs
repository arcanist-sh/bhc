-- `sequence` passed as a value (`fmap sequence`) must run at the monad its
-- type names, not as the list builtin.
newtype R a = R { runR :: Int -> a }

instance Functor R where
  fmap f (R g) = R (f . g)

instance Applicative R where
  pure x = R (const x)
  R f <*> R g = R (\e -> f e (g e))

instance Monad R where
  R g >>= k = R (\e -> runR (k (g e)) e)

item :: Int -> R Int
item k = R (+ k)

collect :: Maybe [R Int] -> Maybe (R [Int])
collect = fmap sequence

main :: IO ()
main = do
  case collect (Just [item 1, item 2, item 3]) of
    Just r -> print (sum (runR r 10))
    Nothing -> putStrLn "none"
  print (sum (runR (sequence [item 4, item 5]) 1))
