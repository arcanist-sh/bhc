-- Kind-annotated binders in data/newtype/class heads (#31).
{-# LANGUAGE KindSignatures #-}
import Data.Kind (Type)

newtype Wrap (f :: Type -> Type) a = Wrap (f a)

class Box (f :: Type -> Type) where
  unbox :: f a -> [a]

instance Box Maybe where
  unbox Nothing = []
  unbox (Just x) = [x]

data Tagged (t :: k) = Tagged Int

data Vec (n :: *) a = Vec [a] deriving Show

unwrap :: Wrap f a -> f a
unwrap (Wrap x) = x

main :: IO ()
main = do
  print (Vec [1, 2 :: Int] :: Vec () Int)
  print (unbox (unwrap (Wrap (Just (3 :: Int)))))
  let Tagged n = (Tagged 7 :: Tagged Maybe)
  print n
