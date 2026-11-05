-- IO functions used as values (#68): a builtin as `>>=`'s continuation, a
-- top-level IO function passed to a higher-order function, and the shape of
-- a generated traversal (`f a >>= \r -> pure (Box r)`).
data Box a = Box a

unbox :: Box a -> a
unbox (Box x) = x

double :: Int -> IO Int
double x = do
  putStrLn (show x)
  return (x * 2)

dbl :: Int -> IO Int
dbl x = return (x * 2)

app :: (Int -> IO Int) -> Int -> IO Int
app f x = f x

applyPrint :: (Int -> IO ()) -> Int -> IO ()
applyPrint f x = f x

tb :: (Int -> IO Int) -> Box Int -> IO (Box Int)
tb f c = case c of
  Box a -> f a >>= \r -> pure (Box r)

main :: IO ()
main = do
  (return 14 :: IO Int) >>= print
  dbl 7 >>= print
  applyPrint print 3
  app dbl 5 >>= print
  app double 6 >>= print
  b <- tb double (Box 21)
  print (unbox b)
  tb dbl (Box 8) >>= \c -> print (unbox c)
