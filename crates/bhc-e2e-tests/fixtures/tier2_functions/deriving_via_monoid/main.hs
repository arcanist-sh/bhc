{-# LANGUAGE DerivingVia #-}
-- DerivingVia through the Data.Monoid/Data.Semigroup newtypes and through a
-- user type's Show (#35).
import Data.Monoid (Sum(..), Product(..), Any(..))
import Data.Semigroup (Max(..))

newtype Score = Score Int deriving (Show) deriving (Semigroup, Monoid) via (Sum Int)
newtype Factor = Factor Int deriving (Show) deriving (Semigroup, Monoid) via (Product Int)
newtype Flag = Flag Bool deriving (Show) deriving (Semigroup, Monoid) via Any
newtype Peak = Peak Int deriving (Show) deriving (Semigroup) via (Max Int)

newtype Shout = Shout String
instance Show Shout where
  show (Shout s) = s ++ "!"
newtype Name = Name String deriving (Show) via Shout

newtype Inner = Inner Int deriving Show
newtype Tag = Tag Int deriving (Show) via Inner

main :: IO ()
main = do
  print (Score 3 <> Score 4)
  print (mconcat [Score 1, Score 2, Score 3], mempty :: Score)
  print (Factor 3 <> Factor 4, mconcat [Factor 2, Factor 5])
  print (Flag False <> Flag True, mempty :: Flag)
  print (Peak 3 <> Peak 9 <> Peak 4)
  print (Name "hi")
  print [Name "a", Name "b"]
  print (Tag 5)
