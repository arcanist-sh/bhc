-- fmap / <$> / <*> at Maybe and Either (#23).
main :: IO ()
main = do
  print ((+) <$> Just 1 <*> Just (2 :: Int))
  print ((+) <$> Just 1 <*> (Nothing :: Maybe Int))
  print ((,) <$> Just 'a' <*> Just True)
  print ((+) <$> (Right 1 :: Either String Int) <*> Right 2)
  print ((+) <$> (Left "e" :: Either String Int) <*> Right 2)
  print (Just (* 3) <*> Just (5 :: Int))
  print (fmap (+1) (Right 1 :: Either String Int))
  print (fmap (+1) (Left "no" :: Either String Int))
  print (fmap length (Right "ab" :: Either Int String))
  print ((* 2) <$> (Right 21 :: Either String Int))
