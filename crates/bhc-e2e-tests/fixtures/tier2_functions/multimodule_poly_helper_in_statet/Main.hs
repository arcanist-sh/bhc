module Main where

import Control.Monad.Reader
import Writer (render)

main :: IO ()
main = do
  r <- runReaderT (render 7) "outer"
  putStrLn r
  r2 <- runReaderT (render 8 >>= \a -> render 9 >>= \b -> return (a ++ " / " ++ b)) "outer"
  putStrLn r2
