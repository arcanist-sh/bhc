-- `<$` at every functor, and Kleisli composition (#71).
import Control.Monad

data Box a = Box a deriving Show

instance Functor Box where
  fmap f (Box x) = Box (f x)

main :: IO ()
main = do
  print (5 <$ Just 'a')
  print (5 <$ [1, 2, 3 :: Int])
  print ((5 :: Int) <$ (Right 'x' :: Either String Char))
  print ((7 :: Int) <$ Box 'y')
  d <- (6 :: Int) <$ return ()
  print d
  (print >=> print) (1 :: Int)
  r <- (return . (+ 1)) >=> (return . (* 2)) $ (3 :: Int)
  print r
  (print <=< (return . (+ 1))) (4 :: Int)
  print ((Just . (+ 1)) >=> (Just . (* 2)) $ (3 :: Int))
  -- `.` binds tighter than `>=>`, which is infixr 1.
  print (mapM (Just . (+ 1) >=> Just . (* 2)) [1, 2 :: Int])
