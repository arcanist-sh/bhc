-- Functions re-exported by Data.Maybe and Data.Either keep their types when
-- imported: typed as placeholders, `print (lefts es)` printed addresses.
import Data.Either (either, lefts, partitionEithers, rights)
import Data.Maybe (catMaybes, fromMaybe, isJust, isNothing, listToMaybe, mapMaybe, maybe, maybeToList)

main :: IO ()
main = do
  let es = [Left "e1", Right (2 :: Int), Left "e2", Right 4]
  print (lefts es, rights es)
  print (partitionEithers es)
  print (either length negate (Left "abc" :: Either String Int))
  print (catMaybes [Just "a", Nothing, Just "c"])
  print (mapMaybe (\w -> if length w > 1 then Just w else Nothing) (words "a bb ccc"))
  print (fromMaybe "none" (listToMaybe (words "first second")))
  print (maybeToList (Just "x"), maybe "?" reverse (Just "olleh"))
  print (isJust (Just "y"), isNothing (Just "y"))
