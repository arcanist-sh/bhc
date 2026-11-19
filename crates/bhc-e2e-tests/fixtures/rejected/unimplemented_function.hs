-- A library function bhc does not implement yet. It compiled with a warning
-- and aborted when reached; a program that compiles must not reach such an
-- abort (#99).
import qualified Data.Text as T

main :: IO ()
main = print (T.chunksOf 2 (T.pack "abcd"))
