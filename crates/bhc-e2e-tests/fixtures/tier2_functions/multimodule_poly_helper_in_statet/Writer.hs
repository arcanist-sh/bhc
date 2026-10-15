-- A `Monad m =>` writer that runs its own StateT layer and calls
-- dictionary-polymorphic helpers inside it (the shape of pandoc's
-- `writeHtml'` calling `metaToContext`).
module Writer (render) where

import Control.Monad.State

data WS = WS { notes :: [Int], html5 :: Bool }

one :: Monad n => n Int
one = return 1

twice :: Monad n => (Int -> n Int) -> Int -> n Int
twice f x = f x >>= f

render :: Monad m => Int -> m String
render k = evalStateT go (WS [k] True)
  where
    go = do
      a <- one
      b <- twice (\v -> return (v + 1)) a
      s <- get
      return (show a ++ " " ++ show b ++ " " ++ show (length (notes s)) ++ " " ++ (if html5 s then "html5" else "html4"))
