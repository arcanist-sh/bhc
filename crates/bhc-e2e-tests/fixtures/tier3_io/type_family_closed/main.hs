{-# LANGUAGE TypeFamilies #-}

-- Closed type family: maps types to their "element" type
type family Elem a where
  Elem [a] = a
  Elem (Maybe a) = a

-- Use the type family in a function signature
getFirst :: [Int] -> Elem [Int]
getFirst (x:_) = x
getFirst [] = 0

-- Another closed type family: Add type-level mapping
type family IsInt a where
  IsInt Int = Bool
  IsInt a = ()

main :: IO ()
main = do
  putStrLn (show (getFirst [10, 20, 30]))
  putStrLn "type families work"
