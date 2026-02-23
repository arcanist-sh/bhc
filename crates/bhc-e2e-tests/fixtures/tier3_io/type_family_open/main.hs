{-# LANGUAGE TypeFamilies #-}

-- Open type family
type family Size a

-- Instances
type instance Size Int = Int
type instance Size [a] = Int

-- Use the type family
intSize :: Int -> Size Int
intSize x = x

listLength :: [a] -> Size [a]
listLength [] = 0
listLength (_:xs) = 1 + listLength xs

main :: IO ()
main = do
  putStrLn (show (intSize 42))
  putStrLn (show (listLength [1, 2, 3, 4, 5]))
  putStrLn "open type families work"
