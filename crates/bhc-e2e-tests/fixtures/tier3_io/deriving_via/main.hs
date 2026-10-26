-- DerivingVia through a user newtype's instances (#35).
{-# LANGUAGE DerivingVia #-}
newtype Add = Add Int deriving Show

instance Semigroup Add where
  Add a <> Add b = Add (a + b)

instance Monoid Add where
  mempty = Add 0

newtype Score = Score Int
  deriving (Show)
  deriving (Semigroup, Monoid) via Add

main :: IO ()
main = do
  print (Score 3 <> Score 4)
  print (mconcat [Score 1, Score 2, Score 3])
  print (mempty :: Score)
