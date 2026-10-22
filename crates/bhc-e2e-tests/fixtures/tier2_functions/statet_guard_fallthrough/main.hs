{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- A guarded case alternative that can fall through to the next one, inside a
-- `MyMonad m => StateT s m` computation (pandoc's `blockToHtmlInner` Para
-- branch). The fallthrough join point's `return` belongs to `StateT s m`.
module Main where
import Control.Monad.State
import Control.Monad.Except

class Monad m => MyMonad m where
  tag :: m Int

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad)

instance MyMonad App where
  tag = return 9

data H = P String | E

runApp :: App a -> IO (Either String a)
runApp a = evalStateT (runExceptT (unApp a)) 0

render :: MyMonad m => Int -> StateT Int m H
render n = do
  c <- return (show n)
  lvl <- get
  case c of
    "" | n > 100 -> return E
    "7" | lvl > 0 -> return (P "seven")
    _ -> return $ P c

go :: MyMonad m => Int -> Int -> m String
go lvl n = do
  h <- evalStateT (render n) lvl
  return (case h of { P s -> s; E -> "E" })

main :: IO ()
main = mapM_ (\(l, n) -> runApp (go l n) >>= putStrLn . either id id) [(0, 5), (1, 7), (0, 7)]
