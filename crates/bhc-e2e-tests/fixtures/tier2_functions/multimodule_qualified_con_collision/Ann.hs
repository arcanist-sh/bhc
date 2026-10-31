module Ann (Table (..), Cell (..), toTable) where

-- Same constructor name as Doc's `Table`, different type and tag.
data Table = Table Int Int

toTable :: Int -> Int -> Table
toTable = Table

data Cell = Cell Int
