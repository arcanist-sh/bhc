-- Cross-transformer test: WriterT over ReaderT
-- Stack: WriterT String (ReaderT String IO) a
-- Uses: tell, lift ask, runWriterT, runReaderT

comp :: WriterT String (ReaderT String IO) String
comp = do
    r <- lift ask
    tell "got: "
    tell r
    return (r ++ "!")

main :: IO ()
main = do
    pair <- runReaderT (runWriterT comp) "hello"
    let result = fst pair
    let log = snd pair
    putStrLn result
    putStrLn log
