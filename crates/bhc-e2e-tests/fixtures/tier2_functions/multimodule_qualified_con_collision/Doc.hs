module Doc (Block (..), Cell (..)) where

data Block = Plain Int | Para Int | Table Int Int

-- Same type AND constructor name as Ann's `Cell`; only the arity differs.
data Cell = Cell Int Int Int
