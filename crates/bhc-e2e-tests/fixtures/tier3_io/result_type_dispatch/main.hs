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

class Mk a where
    mk :: Int -> a

instance Mk Int where
    mk x = x

-- A constrained VALUE. It is never the head of an application, so no argument
-- can drive resolution; the dictionary has to come from this occurrence's own
-- recorded type. Left unresolved it is a bare `\$dMk -> ...` closure, and
-- printing it prints a pointer -- silently, with no warning.
seven :: Mk a => a
seven = mk 7

-- A nullary class method, whose recorded occurrence type pins the class
-- parameter (MyM) but leaves the element type a variable.
class Monad m => Tagged m where
    tag :: m Int

instance Tagged MyM where
    tag = MyM (\s -> (99, s))

main :: IO ()
main = do
    print (seven :: Int)
    print (fst (runMyM tag 0))
