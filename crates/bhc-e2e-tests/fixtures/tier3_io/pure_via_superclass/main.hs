module Main where

newtype MyM a = MyM { runMyM :: Int -> (a, Int) }

instance Functor MyM where
    fmap g (MyM h) = MyM (\s -> let (a, s') = h s in (g a, s'))

instance Applicative MyM where
    pure x = MyM (\s -> (x, s))
    MyM f <*> MyM h = MyM (\s -> let (g, s1) = f s
                                     (a, s2) = h s1
                                 in (g a, s2))

instance Monad MyM where
    return x = MyM (\s -> (x, s))
    MyM h >>= k = MyM (\s -> let (a, s') = h s in runMyM (k a) s')

-- A user class whose superclass chain reaches Applicative.
class Monad m => Tagged m where
    tag :: m Int

instance Tagged MyM where
    tag = MyM (\s -> (99, s))

-- The monad of `return`/`pure` here appears only in the RESULT type, so no
-- argument carries it at runtime and no concrete instance is in sight at the
-- definition site. The enclosing dictionary is the only route to the right
-- `pure`, reached by hopping Tagged -> Monad -> Applicative.
mkVal :: Tagged m => m Int
mkVal = return 7

mkPure :: Tagged m => Int -> m Int
mkPure n = pure (n + 1)

main :: IO ()
main = do
    print (fst (runMyM mkVal 0))
    print (fst (runMyM (mkPure 41) 0))
