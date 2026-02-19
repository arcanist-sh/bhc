{-# LANGUAGE TypeOperators #-}

data a :+: b = Inl a | Inr b

showSum :: (Int :+: String) -> String
showSum (Inl n) = show n
showSum (Inr s) = s

main :: IO ()
main = do
  putStrLn (showSum (Inl 42))
  putStrLn (showSum (Inr "hello"))
