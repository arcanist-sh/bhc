module Types (W(..), S(..), P(..), R(..)) where

newtype W = W { unW :: Int } deriving Show
newtype S a = S { getS :: a } deriving Show
newtype P = P Int deriving Show
data R = R { rf :: Int, rg :: Bool } deriving Show
