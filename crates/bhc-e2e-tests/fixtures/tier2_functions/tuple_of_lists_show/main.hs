-- span/break/splitAt/partition return a 2-tuple of lists ([a],[a]). Printing
-- the runtime result rendered a raw pointer on both backends (the type is
-- erased at the show site). Now recognized as a tuple-of-lists and shown
-- structurally. (partition was also unsynthesized on WASM.)
import Data.List (partition)

main :: IO ()
main = do
    print (span (< 3) [1, 2, 3, 4, 5 :: Int])
    print (break (> 3) [1, 2, 3, 4, 5 :: Int])
    print (splitAt 2 [1, 2, 3, 4, 5 :: Int])
    print (splitAt 0 [1, 2, 3 :: Int])
    print (partition even [1, 2, 3, 4, 5, 6 :: Int])
