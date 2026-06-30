-- scanl/scanl1/scanr/scanr1. These rendered as raw pointers on the native
-- backend (not recognized as list-returning by show), and scanr1 was
-- unsynthesized on WASM; both now agree.
import Data.List (scanl1, scanr1)

main :: IO ()
main = do
    print (scanl (+) 0 [1, 2, 3, 4 :: Int])
    print (scanl1 (+) [1, 2, 3, 4 :: Int])
    print (scanr (+) 0 [1, 2, 3, 4 :: Int])
    print (scanr1 (-) [1, 2, 3, 4 :: Int])
