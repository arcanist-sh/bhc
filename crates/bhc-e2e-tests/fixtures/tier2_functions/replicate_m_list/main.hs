-- `replicateM` in the list monad is every combination; it was typed at IO
-- only and printed nothing (closure model) or an address (eager).
import Control.Monad (replicateM)

main :: IO ()
main = do
  print (replicateM 2 "ab")
  print (replicateM 2 [0, 1 :: Int])
  print (length (replicateM 3 "xyz"))
  xs <- replicateM 2 (return 'z')
  putStrLn xs
