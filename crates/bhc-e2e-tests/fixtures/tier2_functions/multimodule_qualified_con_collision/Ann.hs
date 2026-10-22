module Ann (Table (..), toTable) where

-- Same constructor name as Doc's `Table`, different type and tag.
data Table = Table Int Int

toTable :: Int -> Int -> Table
toTable = Table
