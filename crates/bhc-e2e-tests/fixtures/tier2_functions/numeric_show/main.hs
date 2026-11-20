-- Numeric's showFFloat, showHex and showOct (they were stubs that aborted).
-- showFFloat rounds the shortest decimal digits as GHC's roundTo: half to
-- even when exactly half, so `Just 2` of 2.675 is 2.68 and `Just 1` of 2.25
-- is 2.2.
import Numeric (showFFloat, showHex, showOct)

main :: IO ()
main = do
  putStrLn (showFFloat (Just 2) (3.14159 :: Double) "")
  putStrLn (showFFloat (Just 2) (2.675 :: Double) "")
  putStrLn (showFFloat (Just 1) (2.25 :: Double) "")
  putStrLn (showFFloat (Just 2) (9.999 :: Double) "")
  putStrLn (showFFloat (Just 0) (1.5 :: Double) "")
  putStrLn (showFFloat Nothing (0.01 :: Double) " end")
  putStrLn (showFFloat (Just 3) (fromIntegral (7 :: Int) / 3 :: Double) "")
  putStrLn (showHex (255 :: Int) "" ++ " " ++ showOct (8 :: Int) "")
  print (showFFloat (Just 1) (-0.75 :: Double) "x")
