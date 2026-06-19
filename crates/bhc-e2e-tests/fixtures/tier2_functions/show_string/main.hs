label :: Int -> String
label n = "value = " ++ show n

main :: IO ()
main = do
  putStrLn (label 42)
  putStrLn ("pi ~ " ++ show (3.25 :: Double))
  putStrLn ("ok? " ++ show (1 < 2))
  putStrLn (show (10 :: Int) ++ " and " ++ show (20 :: Int))
