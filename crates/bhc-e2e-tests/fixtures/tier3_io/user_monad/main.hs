-- Test: user-defined monad with do-notation
-- Verifies: dictionary dispatch for >>= on non-builtin monad types

newtype Id a = Id a

runId :: Id a -> a
runId (Id x) = x

instance Functor Id where
    fmap f (Id x) = Id (f x)

instance Applicative Id where
    pure x = Id x
    Id f <*> Id x = Id (f x)

instance Monad Id where
    Id x >>= f = f x
    m >> k = m >>= \_ -> k

-- Use do-notation with the Id monad (no type annotation to avoid typeck issue)
compute = do
    x <- Id 10
    y <- Id 20
    Id (x + y)

main :: IO ()
main = do
    let result = runId compute
    putStrLn (show result)
