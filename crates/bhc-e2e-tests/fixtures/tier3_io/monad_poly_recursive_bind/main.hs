-- A recursive function over `Monad m`, called at IO, passes its recursive
-- call to `>>=`. Under the eager IO model that call was delayed, and the
-- continuation got the thunk's address instead of the result (#65).
cnt :: Monad m => (Int -> m Int) -> Int -> m Int
cnt _ 0 = return 0
cnt g n = cnt g (n - 1) >>= \a -> g a

cntIf :: Monad m => (Int -> m Int) -> Int -> m Int
cntIf g n = if n == 0 then return 0 else cntIf g (n - 1) >>= g

sumTo :: Monad m => Int -> m Int
sumTo 0 = return 0
sumTo n = do
  s <- sumTo (n - 1)
  return (s + n)

-- A polymorphic fold whose accumulator is a recursive call (#64).
foldT :: (a -> b -> b) -> b -> [a] -> b
foldT _ z [] = z
foldT f z (x : xs) = f x (foldT f z xs)

-- The same function at Maybe.
justCount :: Maybe Int
justCount = cnt (\x -> Just (x + 1)) 5

main :: IO ()
main = do
  cnt (\x -> return (x + 1)) 3 >>= print
  cntIf (\x -> return (x * 2 + 1)) 4 >>= print
  sumTo 10 >>= print
  print justCount
  r <- cnt (\x -> do { putStrLn ("step " ++ show x); return (x + 1) }) 2
  print r
  print (foldT (+) 0 [1 .. 10 :: Int])
