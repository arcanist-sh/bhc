{-# LANGUAGE TypeFamilies #-}
-- An associated type reduced before instance resolution (`Num (Elem IntBag)`
-- is `Num Int`), and user class methods named like builtin ones (`empty`)
-- (#34).
class Bag b where
  type Elem b
  empty :: b
  insert :: Elem b -> b -> b
  size :: b -> Int
  toL :: b -> [Elem b]

newtype IntBag = IntBag [Int]

instance Bag IntBag where
  type Elem IntBag = Int
  empty = IntBag []
  insert x (IntBag xs) = IntBag (x : xs)
  size (IntBag xs) = length xs
  toL (IntBag xs) = xs

main :: IO ()
main = do
  print (size (insert 1 (empty :: IntBag)))
  print (toL (insert 2 (insert 3 (empty :: IntBag))))
  print (sum (toL (insert 40 (insert 2 (empty :: IntBag)))))
