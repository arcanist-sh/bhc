-- `f <$> x <*> y` over a user Applicative.
--
-- The functor value is the LAST operand of `<$>`; the first is a plain
-- function. Dispatch used to take the first operand's type, resolving
-- `Functor (Int -> Int -> Int)` — no instance — so `<$>` stayed a builtin
-- fmap and the `Tag` was walked as a list.
module Main where

newtype Tag a = Tag (String, a)

instance Functor Tag where
  fmap f (Tag (s, a)) = Tag ('f' : s, f a)

instance Applicative Tag where
  pure a = Tag ("p", a)
  Tag (s1, f) <*> Tag (s2, a) = Tag (s1 ++ "*" ++ s2, f a)

label :: Tag a -> String
label (Tag (s, _)) = s

value :: Tag a -> a
value (Tag (_, a)) = a

two :: Int -> Int -> Int
two a b = a * 10 + b

main :: IO ()
main = do
  let t = two <$> Tag ("x", 1) <*> Tag ("y", 2)
  putStrLn (label t)
  print (value t)
