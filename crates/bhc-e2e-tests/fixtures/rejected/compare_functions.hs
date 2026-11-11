-- Functions cannot be compared (#106).
main :: IO ()
main = print ((+ 1) == (+ (2 :: Int)), not < not)
