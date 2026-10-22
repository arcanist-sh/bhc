{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- Running `StateT s` over a newtype of ExceptT-over-StateT (pandoc's
-- PandocIO shape): bare `get`/`gets` as the whole computation, `fmap` over
-- the newtype, runner results bound through the newtype's own `>>=`, and
-- the newtype's applicative `<*>` (with a builtin operator under-applied).
module Main where
import Control.Monad.State
import Control.Monad.Except

newtype App a = App { unApp :: ExceptT String (StateT Int IO) a }
  deriving (Functor, Applicative, Monad)

data WS = WS { wsFlag :: Bool, wsCount :: Int }

runApp :: App a -> IO (Either String a)
runApp a = evalStateT (runExceptT (unApp a)) 0

bareGet :: App Int
bareGet = evalStateT get 4

bareGets :: App Int
bareGets = evalStateT (gets wsCount) (WS True 3)

mapped :: App Int
mapped = fmap (+ 1) (evalStateT get 4)

mappedPair :: App Int
mappedPair = fmap fst (runStateT (gets wsCount) (WS True 5))

boundRun :: App Int
boundRun = runStateT (gets wsCount) (WS True 6) >>= \p -> return (fst p * 2)

boundExec :: App Int
boundExec = execStateT (modify (\w -> w { wsCount = 8 })) (WS False 0) >>= \w -> return (wsCount w)

flagged :: App Int
flagged = evalStateT (gets wsFlag >>= \b -> return (if b then 1 else 0)) (WS True 0)

appAp :: App Int
appAp = (+) <$> return 1 <*> return 2

appApLeft :: App Int
appApLeft = (+) <$> return 1 <*> App (throwError "short")

report :: App Int -> IO ()
report act = do
  r <- runApp act
  case r of
    Left e -> putStrLn ("Left " ++ e)
    Right n -> print n

main :: IO ()
main = mapM_ report [bareGet, bareGets, mapped, mappedPair, boundRun, boundExec, flagged, appAp, appApLeft]
