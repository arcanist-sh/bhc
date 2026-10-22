module Doc (Block (..)) where

data Block = Plain Int | Para Int | Table Int Int
