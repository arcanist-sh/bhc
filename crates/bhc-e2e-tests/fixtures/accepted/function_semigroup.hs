-- `<>` on functions: a function is a Semigroup when its result is. Pandoc's
-- Ipynb writer orders keys with `keyOrder [...] <> compare`; the class check
-- rejected that as "No instance for Semigroup (a -> a -> Ordering)".
import Data.List (sortBy)
import Data.Ord (comparing)

main :: IO ()
main = print (sortBy (comparing snd <> compare) [(2, 1), (1, 1), (0, 2 :: Int)])
