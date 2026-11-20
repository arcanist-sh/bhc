-- Text.Printf's printf with a literal format (it was a stub): %d %s %c %f
-- %x %%, precision, width and the - and 0 flags, at IO and at String.
import Text.Printf (printf)

main :: IO ()
main = do
  printf "%d items, %.2f avg, %s\n" (3 :: Int) (2.5 :: Double) "ok"
  printf "[%5d] [%-5d] [%05d]\n" (42 :: Int) (42 :: Int) (-42 :: Int)
  printf "%c%c|%x|%%|%f\n" 'h' 'i' (255 :: Int) (3.14159 :: Double)
  let s = printf "%s=%d" "n" (7 :: Int) :: String
  putStrLn s
  printf "no args\n"
  mapM_ (\(n, v) -> printf "%-4s %6.1f\n" n v) [("a", 1.25 :: Double), ("bb", 10.0)]
