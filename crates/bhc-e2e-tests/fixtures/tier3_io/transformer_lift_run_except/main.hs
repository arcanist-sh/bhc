-- `lift` through ReaderT over StateT, and `runExcept` (#37). Both need the
-- transformers library, so the eager IO model does not run this fixture.
import Control.Monad.Except
import Control.Monad.Reader
import Control.Monad.State

main :: IO ()
main = do
  print =<< runStateT (runReaderT (lift get) (1 :: Int)) (10 :: Int)
  print (runExcept (throwError "e" :: Except String Int))
  print (runExcept (return 3 :: Except String Int))
