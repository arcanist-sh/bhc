-- A generic instance over (Traversable t) maps through t's Functor
-- (pandoc-types' Walkable a (t b) with walk f = fmapDefault (walk f)).
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
import qualified Data.Traversable as T

data I = S String | Sp deriving Show

class W a b where
  walk :: (a -> a) -> b -> b

instance W I I where
  walk f x = f x

instance (Foldable t, Traversable t, W a b) => W a (t b) where
  walk f = T.fmapDefault (walk f)

up :: I -> I
up (S s) = S (s ++ "!")
up x = x

main :: IO ()
main = do
  let xs = walk up [S "a", Sp, S "b"] :: [I]
  print (length xs)
  mapM_ (\i -> putStrLn (case i of { S s -> s; Sp -> "_" })) xs
