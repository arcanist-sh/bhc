-- A class method used at a type with no instance. It compiled with a
-- warning and aborted at run time ("`m` is not implemented by bhc"); a
-- program that compiles must not reach such an abort (#99).
class C a where
  m :: a -> Int

instance C Bool where
  m _ = 1

main :: IO ()
main = print (m "no instance for String")
