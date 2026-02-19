{-# LANGUAGE TypeOperators #-}

data (:*:) a b = Pair a b

fst' :: (:*:) a b -> a
fst' (Pair a _) = a

snd' :: (:*:) a b -> b
snd' (Pair _ b) = b

main :: IO ()
main = do
  let p = Pair 10 20
  print (fst' p)
  print (snd' p)
  print (fst' p + snd' p)
