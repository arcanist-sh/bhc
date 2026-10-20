-- `Monad m =>` / `MonadPlus m =>` functions used at the builtin value monads
-- receive dictionaries for them.
import Control.Monad (MonadPlus, mzero, mplus)
import Data.Char (isSpace)

inc :: Monad m => Int -> m Int
inc x = return (x + 1)

pairUp :: Monad m => m Int -> m Int -> m Int
pairUp a b = do
  x <- a
  y <- b
  return (x * 10 + y)

safeStrRead :: (MonadPlus m, Read a) => String -> m a
safeStrRead s = case reads s of
  (d, x) : _ | all isSpace x -> return d
  _ -> mzero

orZero :: MonadPlus m => m Int -> m Int
orZero a = a `mplus` return 0

main :: IO ()
main = do
  print (maybe 0 id (inc 41))
  print (sum (inc 1 :: [Int]))
  print (maybe 0 id (pairUp (Just 4) (Just 2)))
  print (maybe (-1) id (pairUp (Just 4) Nothing))
  print (sum (pairUp [1, 2] [3, 4]))
  print (maybe (-1) id (safeStrRead "42" :: Maybe Int))
  print (maybe (-1) id (safeStrRead "x" :: Maybe Int))
  print (length (safeStrRead "7" :: [Int]))
  print (maybe (-1) id (orZero Nothing))
  print (sum (orZero [5]), length (orZero [5]))
