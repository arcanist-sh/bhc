-- Test: foreign import ccall
-- Verifies: parsing, lowering, and codegen for foreign import declarations
foreign import ccall "sin" c_sin :: Double -> Double
foreign import ccall "cos" c_cos :: Double -> Double
foreign import ccall unsafe "sqrt" c_sqrt :: Double -> Double

main :: IO ()
main = do
  putStrLn (show (c_sin 0.0))
  putStrLn (show (c_cos 0.0))
  putStrLn (show (c_sqrt 4.0))
