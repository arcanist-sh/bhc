{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
-- | Classes whose methods have defaults, for a module that imports them.
module Walk where

data Inline = Str String | Emph [Inline] deriving (Show, Eq)

-- | One parameter; the default comes first in the class.
class Label a where
  label :: a -> String
  label x = "<" ++ name x ++ ">"
  name :: a -> String

instance Label Int where
  name n = show n

instance Label Bool where
  name b = if b then "yes" else "no"
  label b = "!" ++ name b

-- | Two parameters, as in pandoc-types' @Walkable@.
class Walkable a b where
  walk :: (a -> a) -> b -> b
  walk f = fst . walkWith (\x -> (f x, ()))
  walkWith :: (a -> (a, ())) -> b -> (b, ())

instance Walkable Inline Inline where
  walkWith f (Str s) = f (Str s)
  walkWith f (Emph xs) = f (Emph (map (fst . walkWith f) xs))

instance Walkable Inline [Inline] where
  walkWith f xs = (map (fst . walkWith f) xs, ())
