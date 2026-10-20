-- A `MonadPlus m =>` function used at Maybe and at lists, directly and
-- through mapM (pandoc's `isJust (mapM toTaskListItem items)`).
import Control.Monad (MonadPlus, mzero)
import Data.Maybe (isJust, fromMaybe)

task :: MonadPlus m => String -> m Bool
task ('x' : _) = pure True
task _ = mzero

main :: IO ()
main = do
  print (isJust (task "xa" :: Maybe Bool))
  print (isJust (task "a" :: Maybe Bool))
  print (fromMaybe False (task "xa" :: Maybe Bool))
  print (length (task "xa" :: [Bool]))
  print (length (task "a" :: [Bool]))
  print (isJust (mapM task ["a"] :: Maybe [Bool]))
  print (isJust (mapM task ["xa", "xb"] :: Maybe [Bool]))
  print (isJust (mapM task ["xa", "b"] :: Maybe [Bool]))
