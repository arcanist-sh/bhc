-- A constrained function whose signature uses a type SYNONYM that drops a
-- parameter (`type Q s = R s ()`), applied to a value of the expanded type.
-- The dictionary must resolve at the synonym-expanded type, or it is
-- omitted and every later argument shifts by one.
class Source s where
  firstOf :: s -> Int

instance Source [a] where
  firstOf xs = length xs

newtype R s u = R { runR :: s -> u -> Int }

type Q s = R s ()

runQ :: Source s => Q s -> s -> Int
runQ r s = runR r s () + firstOf s

counter :: R [Int] ()
counter = R (\s _ -> sum s)

main :: IO ()
main = do
  print (runQ counter [1, 2, 3])
  print (runQ counter [10, 20])
