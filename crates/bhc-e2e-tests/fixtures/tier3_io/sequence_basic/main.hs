import qualified Data.Sequence as Seq

-- Helper functions to avoid qualified operator syntax (Seq.<| etc.)
-- which the parser doesn't support yet
seqCons :: a -> [a] -> [a]
seqCons = (:)

main :: IO ()
main = do
    -- empty + length
    putStrLn (show (Seq.length Seq.empty))

    -- singleton + length
    putStrLn (show (Seq.length (Seq.singleton 42)))

    -- fromList + length
    putStrLn (show (Seq.length (Seq.fromList [1, 2, 3, 4, 5])))

    -- index
    putStrLn (show (Seq.index (Seq.fromList [10, 20, 30]) 1))

    -- toList roundtrip
    putStrLn (show (Seq.toList (Seq.fromList [10, 20, 30])))

    -- take/drop
    putStrLn (show (Seq.length (Seq.take 2 (Seq.fromList [1, 2, 3, 4]))))
    putStrLn (show (Seq.length (Seq.drop 2 (Seq.fromList [1, 2, 3, 4]))))

    -- reverse
    putStrLn (show (Seq.toList (Seq.reverse (Seq.fromList [1, 2, 3]))))

    -- null
    putStrLn (show (Seq.null Seq.empty))
    putStrLn (show (Seq.null (Seq.singleton 1)))
