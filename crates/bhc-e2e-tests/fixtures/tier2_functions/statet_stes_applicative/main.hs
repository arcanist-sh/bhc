{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- Functor/Applicative ops of `StateT s` over an ExceptT-over-StateT monad
-- (pandoc's writer stack), both written at the concrete newtype and reached
-- through a `MyMonad m =>` function specialized at it.
module Main where
import Control.Monad.State
import Control.Monad.Except

class Monad m => MyMonad m where
  tag :: m Int

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad, MonadError String)

instance MyMonad App where
  tag = return 9

runApp :: App a -> IO (Either String a)
runApp a = evalStateT (runExceptT (unApp a)) 0

viaFmap :: Int -> App String
viaFmap st = evalStateT (fmap show get) st

viaAp :: Int -> App String
viaAp st = evalStateT ((++) <$> return "p" <*> fmap show get) st

shortCircuit :: Int -> App String
shortCircuit st = evalStateT (fmap (++ "!") (lift (throwError "boom"))) st

poly :: MyMonad m => Int -> m String
poly st = evalStateT (concat <$> mapM (\t -> return t) ["a", "b"]) st

-- `<$>`/`<*>` at the polymorphic monad itself, reached in the specialized
-- clone as builtin operators used as VALUES (pandoc's `fst <$> evalStateT …`).
polyFst :: MyMonad m => Int -> m String
polyFst n = (show . fst) <$> evalStateT (return (n, "x")) 0

polyAp :: MyMonad m => Int -> m String
polyAp n = (\a b -> show (a + b)) <$> return n <*> evalStateT (return 10) 0

report :: Either String String -> IO ()
report = either (putStrLn . ("Left " ++)) putStrLn

main :: IO ()
main = do
  runApp (viaFmap 42) >>= report
  runApp (viaAp 7) >>= report
  runApp (shortCircuit 0) >>= report
  runApp (poly 0) >>= report
  runApp (polyFst 4) >>= report
  runApp (polyAp 5) >>= report
