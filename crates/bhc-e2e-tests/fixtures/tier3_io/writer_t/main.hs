import Control.Monad.Writer

main = do
    pair <- runWriterT computation
    putStrLn (fst pair)
    putStrLn (snd pair)

computation :: WriterT String IO String
computation = do
    tell "hello "
    tell "world"
    return "done"
