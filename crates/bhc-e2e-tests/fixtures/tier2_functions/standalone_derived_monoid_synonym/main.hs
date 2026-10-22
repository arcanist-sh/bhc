{-# LANGUAGE StandaloneDeriving, GeneralizedNewtypeDeriving, FlexibleInstances #-}
-- `deriving instance Monoid Blocks` where `Blocks` is a synonym of a newtype
-- (pandoc-types' Builder): the derived methods reuse the field's instance.
module Main where

newtype Many a = Many { unMany :: [a] }
type Blocks = Many Int

deriving instance Semigroup Blocks
deriving instance Monoid Blocks

cat :: Monoid m => [m] -> m
cat = foldr (<>) mempty

main :: IO ()
main = do
  print (length (unMany (mempty :: Blocks)))
  print (sum (unMany (Many [1, 2] <> Many [3] :: Blocks)))
  print (sum (unMany (mconcat [Many [4], Many [5], mempty] :: Blocks)))
  print (length (unMany (cat [Many [1], mempty, Many [2, 3]] :: Blocks)))
