main :: IO ()
main = do
  print [1, 2, 3 :: Int]
  print (Just (5 :: Int))
  print (Nothing :: Maybe Int)
  print (1 :: Int, True)
  print [True, False]
  putStrLn (show (Just (Just (7 :: Int))))
  print (Left 3 :: Either Int Bool)
