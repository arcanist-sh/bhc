-- find/elemIndex/lookup return Maybe; show/print of their runtime results was
-- rendering a raw pointer on both backends (the runtime Maybe-show path was
-- only wired for readMaybe / partially-applied calls). The Just field is shown
-- as Int, the common case.
import Data.List (find, elemIndex, findIndex)

main :: IO ()
main = do
    print (find even [1, 3, 4, 5, 6 :: Int])
    print (find (> 10) [1, 2, 3 :: Int])
    print (elemIndex 3 [1, 2, 3, 4 :: Int])
    print (elemIndex 9 [1, 2, 3 :: Int])
    print (findIndex (> 2) [1, 2, 3, 4 :: Int])
    print (findIndex (> 9) [1, 2, 3 :: Int])
    print (lookup 2 [(1, 10), (2, 20), (3, 30) :: (Int, Int)])
    print (lookup 5 [(1, 10), (2, 20) :: (Int, Int)])
