-- `Con <$> action` in an instance method that has its own `Functor m`: the
-- operator must come from that dictionary, as `fmap` does. It stayed the
-- builtin and walked the action as a list.
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, FlexibleContexts #-}
import Data.Functor.Identity (Identity (runIdentity))
import qualified Data.Traversable as T
import qualified Data.Foldable as F

data Inl = Str String | Emph [Inl] deriving (Show, Eq)
data Cap = Cap [Blk] deriving (Show, Eq)
data Blk = Plain [Inl] | Quote [Blk] | Rule | Code String | Fig Cap [Blk] deriving (Show, Eq)

class Walkable a b where
  walk  :: (a -> a) -> b -> b
  walk f = runIdentity . walkM (return . f)
  walkM :: (Monad m, Applicative m, Functor m) => (a -> m a) -> b -> m b
  query :: Monoid c => (a -> c) -> b -> c

instance (Foldable t, Traversable t, Walkable a b) => Walkable a (t b) where
  walk f  = T.fmapDefault (walk f)
  walkM f = T.mapM (walkM f)
  query f = F.foldMap (query f)

instance Walkable Inl Inl where
  walkM f x = walkInlM f x >>= f
  query f x = f x <> queryInl f x

instance Walkable Inl Blk where
  walkM = walkBlkM
  query = queryBlk

instance Walkable Inl Cap where
  walkM f (Cap bs) = Cap <$> walkM f bs
  query f (Cap bs) = query f bs

walkInlM :: (Walkable a [Inl], Monad m, Applicative m, Functor m) => (a -> m a) -> Inl -> m Inl
walkInlM _ (Str s) = return (Str s)
walkInlM f (Emph xs) = Emph <$> walkM f xs

queryInl :: (Walkable a [Inl], Monoid c) => (a -> c) -> Inl -> c
queryInl _ (Str _) = mempty
queryInl f (Emph xs) = query f xs

walkBlkM :: (Walkable a [Blk], Walkable a [Inl], Walkable a Cap, Monad m, Applicative m, Functor m) => (a -> m a) -> Blk -> m Blk
walkBlkM f (Plain xs) = Plain <$> walkM f xs
walkBlkM f (Quote bs) = Quote <$> walkM f bs
walkBlkM _ x@Code {} = return x
walkBlkM _ Rule = return Rule
walkBlkM f (Fig capt blks)
  = do capt' <- walkM f capt
       blks' <- walkM f blks
       return $ Fig capt' blks'

queryBlk :: (Walkable a [Blk], Walkable a [Inl], Walkable a Cap, Monoid c) => (a -> c) -> Blk -> c
queryBlk f (Plain xs) = query f xs
queryBlk f (Quote bs) = query f bs
queryBlk _ Rule = mempty
queryBlk _ (Code _) = mempty
queryBlk f (Fig c bs) = query f c <> query f bs

fx :: Inl -> Inl
fx (Str x) = Str (x ++ "!")
fx x = x

main :: IO ()
main = do
  let b = Plain [Str "img", Emph [Str "e"]]
  print (walk fx b)
  print (walk fx [Quote [b], Fig (Cap [b]) [b, Rule, Code "c"]])
  print (walk fx (Cap [b]))
  print (walk fx (Box3 [Str "k"]))
  print (walk fx (Box [Str "k"]))
  print (walk fx (Box2 [Str "k"]))

data Box = Box [Inl] deriving Show
instance Walkable Inl Box where
  walkM f (Box xs) = fmap Box (walkM f xs)
  query f (Box xs) = query f xs
data Box2 = Box2 [Inl] deriving Show
instance Walkable Inl Box2 where
  walkM f (Box2 xs) = Box2 <$> walkM f xs
  query f (Box2 xs) = query f xs
data Box3 = Box3 [Inl] deriving Show
instance Walkable Inl Box3 where
  walkM f (Box3 xs) = walkM f xs >>= \ys -> return (Box3 ys)
  query f (Box3 xs) = query f xs
