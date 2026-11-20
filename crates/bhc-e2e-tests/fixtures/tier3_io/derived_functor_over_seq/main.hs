-- A derived Functor for a newtype over `Seq`: `fmap` maps the sequence. It
-- was compiled as an action in the ambient monad and returned a closure.
{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
import Data.Sequence (Seq)
import qualified Data.Sequence as Seq
import qualified Data.Foldable as F

newtype Many a = Many { unMany :: Seq a }
  deriving (Eq, Ord, Foldable, Traversable, Functor, Show)

fromList :: [a] -> Many a
fromList = Many . Seq.fromList

main :: IO ()
main = do
  let m = fromList [1, 2, 3 :: Int]
  print (F.toList (fmap (* 2) m))
  print (F.toList ((+ 1) <$> m))
  print (F.toList (fmap show m))
  print (length (fmap (* 2) m))
  print (F.toList (fmap (* 2) (fromList [])))
