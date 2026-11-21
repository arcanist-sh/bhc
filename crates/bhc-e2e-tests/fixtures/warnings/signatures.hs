-- Top-level bindings without signatures: -Wall reports each with the type
-- bhc inferred, in Haskell's notation.
import qualified Data.Map as M

pair a b = (a, b)

lookupOr d k m = M.findWithDefault d k m

main :: IO ()
main = print (pair (lookupOr 0 "k" (M.fromList [("k", 1 :: Int)])) True)
