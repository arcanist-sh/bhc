-- Data.List.unfoldr returns a list, but print rendered it as a raw pointer on
-- both backends (not recognized as list-returning by show). The value was
-- always correct (length/sum worked); only the show dispatch was missing.
import Data.List (unfoldr)

main :: IO ()
main = do
    print (unfoldr (\n -> if n > 3 then Nothing else Just (n, n + 1)) (1 :: Int))
    print (unfoldr (\n -> if n > 100 then Nothing else Just (n, n * 2)) (1 :: Int))
    print (unfoldr (\n -> if n > 0 then Nothing else Just (n, n + 1)) (5 :: Int))
