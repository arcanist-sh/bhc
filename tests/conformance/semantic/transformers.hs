-- Test: transformers
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Phase 5 - Monad Transformers

{-# HASKELL_EDITION 2026 #-}

module TransformersTest where

import BHC.Control.Monad.Trans
import BHC.Control.Monad.Identity
import BHC.Control.Monad.Reader
import BHC.Control.Monad.Writer
import BHC.Control.Monad.State
import BHC.Control.Monad.Except
import BHC.Control.Monad.Maybe
import BHC.Control.Monad.RWS
import BHC.Control.Monad.Cont

-- ================================================================
-- Identity Tests
-- ================================================================

testIdentityFunctor :: Bool
testIdentityFunctor =
    fmap (+1) (Identity 5) == Identity 6
-- Result: True

testIdentityApplicative :: Bool
testIdentityApplicative =
    (Identity (+1) <*> Identity 5) == Identity 6 &&
    pure 42 == Identity 42
-- Result: True

testIdentityMonad :: Bool
testIdentityMonad =
    (Identity 5 >>= \x -> Identity (x + 1)) == Identity 6
-- Result: True

testRunIdentity :: Bool
testRunIdentity =
    runIdentity (Identity 42) == 42
-- Result: True

testIdentityT :: Bool
testIdentityT =
    runIdentity (runIdentityT (IdentityT (Identity 42))) == 42
-- Result: True

-- ================================================================
-- Reader Tests
-- ================================================================

testReaderAsk :: Bool
testReaderAsk =
    runReader ask 10 == 10
-- Result: True

testReaderAsks :: Bool
testReaderAsks =
    runReader (asks (*2)) 10 == 20
-- Result: True

testReaderLocal :: Bool
testReaderLocal =
    runReader (local (*2) ask) 10 == 20
-- Result: True

testReaderNested :: Bool
testReaderNested =
    let comp = do
            x <- ask
            y <- local (+1) ask
            return (x, y)
    in runReader comp 10 == (10, 11)
-- Result: True

testReaderT :: Bool
testReaderT =
    let comp = do
            x <- ask
            lift (Just (x * 2))
    in runReaderT comp 5 == Just 10
-- Result: True

testReaderMonadReader :: Bool
testReaderMonadReader =
    runReader (readerM (*3)) 10 == 30
-- Result: True

-- ================================================================
-- Writer Tests
-- ================================================================

testWriterTell :: Bool
testWriterTell =
    runWriter (tell "hello") == ((), "hello")
-- Result: True

testWriterListen :: Bool
testWriterListen =
    let (result, output) = runWriter $ do
            tell "a"
            (x, w) <- listen (tell "b" >> return 1)
            tell "c"
            return (x, w)
    in result == (1, "b") && output == "abc"
-- Result: True

testWriterPass :: Bool
testWriterPass =
    let (result, output) = runWriter $ do
            tell "a"
            pass $ do
                tell "b"
                return (1, reverse)
            tell "c"
    in result == 1 && output == "abc"  -- "b" reversed to "b", then concatenated
-- Result: True

testWriterCensor :: Bool
testWriterCensor =
    let (result, output) = runWriter $ do
            tell "a"
            censor (map toUpper) (tell "b" >> return 1)
            tell "c"
    in result == 1 && output == "aBc"
-- Result: True

testWriterListens :: Bool
testWriterListens =
    let (result, output) = runWriter $ listens length (tell "hello")
    in result == ((), 5) && output == "hello"
-- Result: True

-- ================================================================
-- State Tests
-- ================================================================

testStateGet :: Bool
testStateGet =
    evalState get 42 == 42
-- Result: True

testStatePut :: Bool
testStatePut =
    execState (put 100) 0 == 100
-- Result: True

testStateModify :: Bool
testStateModify =
    execState (modify (*2)) 21 == 42
-- Result: True

testStateGets :: Bool
testStateGets =
    evalState (gets show) 42 == "42"
-- Result: True

testStateNested :: Bool
testStateNested =
    let comp = do
            x <- get
            put (x + 1)
            y <- get
            modify (*2)
            z <- get
            return (x, y, z)
    in runState comp 10 == ((10, 11, 22), 22)
-- Result: True

testStateT :: Bool
testStateT =
    let comp = do
            x <- get
            lift (Just ())
            put (x + 1)
            return x
    in runStateT comp 10 == Just (10, 11)
-- Result: True

testStateMonadState :: Bool
testStateMonadState =
    let comp = stateS (\s -> (s * 2, s + 1))
    in runState comp 10 == (20, 11)
-- Result: True

-- ================================================================
-- Except Tests
-- ================================================================

testExceptThrow :: Bool
testExceptThrow =
    runExcept (throwE "error" :: Except String Int) == Left "error"
-- Result: True

testExceptReturn :: Bool
testExceptReturn =
    runExcept (return 42 :: Except String Int) == Right 42
-- Result: True

testExceptCatch :: Bool
testExceptCatch =
    let comp = throwE "error" `catchE` \e -> return (length e)
    in runExcept comp == Right 5
-- Result: True

testExceptSequence :: Bool
testExceptSequence =
    let comp = do
            x <- return 1
            y <- return 2
            return (x + y)
    in runExcept (comp :: Except String Int) == Right 3
-- Result: True

testExceptFailure :: Bool
testExceptFailure =
    let comp = do
            x <- return 1
            _ <- throwE "oops"
            return (x + 1)  -- Never reached
    in runExcept (comp :: Except String Int) == Left "oops"
-- Result: True

testExceptT :: Bool
testExceptT =
    let comp = do
            x <- lift (Just 1)
            y <- lift (Just 2)
            return (x + y)
    in runExceptT (comp :: ExceptT String Maybe Int) == Just (Right 3)
-- Result: True

testWithExceptT :: Bool
testWithExceptT =
    let comp = throwE "error" :: Except String Int
        mapped = withExcept length comp
    in runExcept mapped == Left 5
-- Result: True

-- ================================================================
-- Maybe Tests
-- ================================================================

testMaybeSuccess :: Bool
testMaybeSuccess =
    runMaybeT (return 42 :: MaybeT Identity Int) == Identity (Just 42)
-- Result: True

testMaybeFailure :: Bool
testMaybeFailure =
    runMaybeT (empty :: MaybeT Identity Int) == Identity Nothing
-- Result: True

testMaybeSequence :: Bool
testMaybeSequence =
    let comp = do
            x <- return 1
            y <- return 2
            return (x + y)
    in runIdentity (runMaybeT comp) == Just 3
-- Result: True

testMaybeShortCircuit :: Bool
testMaybeShortCircuit =
    let comp = do
            x <- return 1
            _ <- MaybeT (Identity Nothing)
            return (x + 1)  -- Never reached
    in runIdentity (runMaybeT comp) == Nothing
-- Result: True

testMaybeAlternative :: Bool
testMaybeAlternative =
    let comp = empty <|> return 42
    in runIdentity (runMaybeT comp) == Just 42
-- Result: True

testHoistMaybe :: Bool
testHoistMaybe =
    runIdentity (runMaybeT (hoistMaybe (Just 42))) == Just 42 &&
    runIdentity (runMaybeT (hoistMaybe Nothing)) == Nothing
-- Result: True

-- ================================================================
-- RWS Tests
-- ================================================================

testRWSBasic :: Bool
testRWSBasic =
    let comp = do
            r <- ask
            s <- get
            tell [r, s]
            put (s + 1)
            return (r * s)
    in runRWS comp 2 3 == (6, 4, [2, 3])
-- Result: True

testRWSEval :: Bool
testRWSEval =
    let comp = do
            r <- ask
            tell [r]
            return (r * 2)
    in evalRWS comp 5 0 == (10, [5])
-- Result: True

testRWSExec :: Bool
testRWSExec =
    let comp = do
            tell ["log"]
            modify (+1)
    in execRWS comp () 0 == (1, ["log"])
-- Result: True

testRWSLocal :: Bool
testRWSLocal =
    let comp = do
            x <- ask
            y <- local (*2) ask
            return (x, y)
    in evalRWS comp 5 0 == ((5, 10), [])
-- Result: True

testRWSCensor :: Bool
testRWSCensor =
    let comp = censor (map toUpper) (tell "hello")
    in execRWS comp () 0 == (0, "HELLO")
-- Result: True

-- ================================================================
-- Cont Tests
-- ================================================================

testContBasic :: Bool
testContBasic =
    evalCont (return 42) == 42
-- Result: True

testContBind :: Bool
testContBind =
    let comp = do
            x <- return 1
            y <- return 2
            return (x + y)
    in evalCont comp == 3
-- Result: True

testContCallCC :: Bool
testContCallCC =
    let comp = callCC $ \k -> do
            _ <- k 10      -- Early return
            return 20      -- Never reached
    in evalCont comp == 10
-- Result: True

testContCallCCNoExit :: Bool
testContCallCCNoExit =
    let comp = callCC $ \_ -> do
            return 20      -- Normal return
    in evalCont comp == 20
-- Result: True

testContReset :: Bool
testContReset =
    let comp = reset $ do
            x <- shift $ \k -> do
                a <- k 1
                b <- k 2
                return (a + b)
            return (x * 10)
    in evalCont comp == 30  -- (1*10) + (2*10) = 30
-- Result: True

-- ================================================================
-- Transformer Stacking Tests
-- ================================================================

testReaderState :: Bool
testReaderState =
    let comp :: ReaderT Int (State Int) Int
        comp = do
            r <- ask
            s <- lift get
            lift (put (s + r))
            return (r * s)
    in runState (runReaderT comp 2) 3 == (6, 5)
-- Result: True

testStateWriter :: Bool
testStateWriter =
    let comp :: StateT Int (Writer String) Int
        comp = do
            s <- get
            lift (tell (show s))
            put (s + 1)
            return s
    in runWriter (runStateT comp 10) == ((10, 11), "10")
-- Result: True

testExceptState :: Bool
testExceptState =
    let comp :: ExceptT String (State Int) Int
        comp = do
            s <- lift get
            if s > 0
                then do
                    lift (put (s - 1))
                    return s
                else throwE "negative"
    in runState (runExceptT comp) 5 == (Right 5, 4)
-- Result: True

testMaybeReader :: Bool
testMaybeReader =
    let comp :: MaybeT (Reader Int) Int
        comp = do
            r <- lift ask
            if r > 0
                then return r
                else empty
    in runReader (runMaybeT comp) 10 == Just 10 &&
       runReader (runMaybeT comp) (-1) == Nothing
-- Result: True

-- ================================================================
-- Monad Laws Tests
-- ================================================================

-- Left identity: return a >>= f  ≡  f a
testLeftIdentity :: Bool
testLeftIdentity =
    let f x = Identity (x + 1)
    in (return 5 >>= f) == f 5
-- Result: True

-- Right identity: m >>= return  ≡  m
testRightIdentity :: Bool
testRightIdentity =
    let m = Identity 5
    in (m >>= return) == m
-- Result: True

-- Associativity: (m >>= f) >>= g  ≡  m >>= (\x -> f x >>= g)
testAssociativity :: Bool
testAssociativity =
    let m = Identity 5
        f x = Identity (x + 1)
        g x = Identity (x * 2)
    in ((m >>= f) >>= g) == (m >>= (\x -> f x >>= g))
-- Result: True

-- ================================================================
-- Edge Cases
-- ================================================================

testEmptyWriter :: Bool
testEmptyWriter =
    runWriter (return 42 :: Writer String Int) == (42, "")
-- Result: True

testStateNoChange :: Bool
testStateNoChange =
    runState (return 42) 0 == (42, 0)
-- Result: True

testNestedCallCC :: Bool
testNestedCallCC =
    let comp = callCC $ \k1 ->
            callCC $ \k2 -> do
                _ <- k1 10
                k2 20
    in evalCont comp == 10
-- Result: True

testLiftIO :: Bool
testLiftIO =
    -- Can't directly test IO, but we can check the types work
    let comp :: ReaderT Int IO ()
        comp = do
            _ <- ask
            liftIO (return ())
    in True  -- Type checks
-- Result: True

-- ================================================================
-- Main
-- ================================================================

main :: IO ()
main = do
    -- Identity
    print testIdentityFunctor
    print testIdentityApplicative
    print testIdentityMonad
    print testRunIdentity
    print testIdentityT

    -- Reader
    print testReaderAsk
    print testReaderAsks
    print testReaderLocal
    print testReaderNested
    print testReaderT
    print testReaderMonadReader

    -- Writer
    print testWriterTell
    print testWriterListen
    print testWriterPass
    print testWriterCensor
    print testWriterListens

    -- State
    print testStateGet
    print testStatePut
    print testStateModify
    print testStateGets
    print testStateNested
    print testStateT
    print testStateMonadState

    -- Except
    print testExceptThrow
    print testExceptReturn
    print testExceptCatch
    print testExceptSequence
    print testExceptFailure
    print testExceptT
    print testWithExceptT

    -- Maybe
    print testMaybeSuccess
    print testMaybeFailure
    print testMaybeSequence
    print testMaybeShortCircuit
    print testMaybeAlternative
    print testHoistMaybe

    -- RWS
    print testRWSBasic
    print testRWSEval
    print testRWSExec
    print testRWSLocal
    print testRWSCensor

    -- Cont
    print testContBasic
    print testContBind
    print testContCallCC
    print testContCallCCNoExit
    print testContReset

    -- Stacking
    print testReaderState
    print testStateWriter
    print testExceptState
    print testMaybeReader

    -- Laws
    print testLeftIdentity
    print testRightIdentity
    print testAssociativity

    -- Edge cases
    print testEmptyWriter
    print testStateNoChange
    print testNestedCallCC
    print testLiftIO
