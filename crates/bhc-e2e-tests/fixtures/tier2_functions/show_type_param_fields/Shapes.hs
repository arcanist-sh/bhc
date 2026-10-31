module Shapes (Box(..)) where

newtype Box a = Box { unBox :: a } deriving Show
