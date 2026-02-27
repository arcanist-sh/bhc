data Box a = Box a

instance Functor Box where
    fmap f (Box x) = Box (f x)

instance Applicative Box where
    pure x = Box x
    Box f <*> Box x = Box (f x)

instance Monad Box where
    Box x >>= f = f x
    m >> k = m >>= \_ -> k

compute = do
    x <- Box 42
    Box x

main :: IO ()
main = do
    case compute of
        Box x -> putStrLn (show x)
