-- |
-- Module      : BHC.System.IO
-- Description : Input/Output operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.System.IO (
    -- * The IO monad
    IO,
    
    -- * Standard handles
    Handle,
    stdin, stdout, stderr,
    
    -- * Opening and closing
    openFile,
    hClose,
    withFile,
    
    -- * IO modes
    IOMode(..),
    
    -- * Reading
    hGetChar, hGetLine, hGetContents,
    hLookAhead, hReady,
    
    -- * Writing
    hPutChar, hPutStr, hPutStrLn, hPrint,
    hFlush,
    
    -- * Seeking
    hSeek, hTell,
    SeekMode(..),
    
    -- * Queries
    hIsEOF, hIsOpen, hIsClosed,
    hIsReadable, hIsWritable, hIsSeekable,
    
    -- * Buffering
    BufferMode(..),
    hSetBuffering, hGetBuffering,
    
    -- * File operations
    readFile, writeFile, appendFile,
    
    -- * Console I/O
    putChar, putStr, putStrLn, print,
    getChar, getLine, getContents,
    interact,
    
    -- * Errors
    IOError, ioError, userError,
    catch, try,
) where

import BHC.Prelude hiding (
    putChar, putStr, putStrLn, print,
    getChar, getLine, getContents, interact,
    readFile, writeFile, appendFile
    )

-- | A file handle.
data Handle

-- | File opening modes.
data IOMode
    = ReadMode
    | WriteMode
    | AppendMode
    | ReadWriteMode
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | File seek modes.
data SeekMode
    = AbsoluteSeek
    | RelativeSeek
    | SeekFromEnd
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Buffer modes.
data BufferMode
    = NoBuffering
    | LineBuffering
    | BlockBuffering (Maybe Int)
    deriving (Eq, Ord, Show, Read)

-- Standard handles
foreign import ccall "bhc_stdin" stdin :: Handle
foreign import ccall "bhc_stdout" stdout :: Handle
foreign import ccall "bhc_stderr" stderr :: Handle

-- Opening and closing
foreign import ccall "bhc_open_file"
    openFile :: FilePath -> IOMode -> IO Handle

foreign import ccall "bhc_close_handle"
    hClose :: Handle -> IO ()

withFile :: FilePath -> IOMode -> (Handle -> IO a) -> IO a
withFile path mode action = do
    h <- openFile path mode
    r <- action h `catch` \e -> hClose h >> throw e
    hClose h
    return r

-- Reading
foreign import ccall "bhc_hGetChar" hGetChar :: Handle -> IO Char
foreign import ccall "bhc_hGetLine" hGetLine :: Handle -> IO String
foreign import ccall "bhc_hGetContents" hGetContents :: Handle -> IO String
foreign import ccall "bhc_hLookAhead" hLookAhead :: Handle -> IO Char
foreign import ccall "bhc_hReady" hReady :: Handle -> IO Bool

-- Writing
foreign import ccall "bhc_hPutChar" hPutChar :: Handle -> Char -> IO ()
foreign import ccall "bhc_hPutStr" hPutStr :: Handle -> String -> IO ()

hPutStrLn :: Handle -> String -> IO ()
hPutStrLn h s = hPutStr h s >> hPutChar h '\n'

hPrint :: Show a => Handle -> a -> IO ()
hPrint h x = hPutStrLn h (show x)

foreign import ccall "bhc_hFlush" hFlush :: Handle -> IO ()

-- Seeking
foreign import ccall "bhc_hSeek" hSeek :: Handle -> SeekMode -> Integer -> IO ()
foreign import ccall "bhc_hTell" hTell :: Handle -> IO Integer

-- Queries
foreign import ccall "bhc_hIsEOF" hIsEOF :: Handle -> IO Bool
foreign import ccall "bhc_hIsOpen" hIsOpen :: Handle -> IO Bool
foreign import ccall "bhc_hIsClosed" hIsClosed :: Handle -> IO Bool
foreign import ccall "bhc_hIsReadable" hIsReadable :: Handle -> IO Bool
foreign import ccall "bhc_hIsWritable" hIsWritable :: Handle -> IO Bool
foreign import ccall "bhc_hIsSeekable" hIsSeekable :: Handle -> IO Bool

-- Buffering
foreign import ccall "bhc_hSetBuffering" hSetBuffering :: Handle -> BufferMode -> IO ()
foreign import ccall "bhc_hGetBuffering" hGetBuffering :: Handle -> IO BufferMode

-- File operations
foreign import ccall "bhc_readFile" readFile :: FilePath -> IO String
foreign import ccall "bhc_writeFile" writeFile :: FilePath -> String -> IO ()
foreign import ccall "bhc_appendFile" appendFile :: FilePath -> String -> IO ()

-- Console I/O
putChar :: Char -> IO ()
putChar = hPutChar stdout

putStr :: String -> IO ()
putStr = hPutStr stdout

putStrLn :: String -> IO ()
putStrLn = hPutStrLn stdout

print :: Show a => a -> IO ()
print = hPrint stdout

getChar :: IO Char
getChar = hGetChar stdin

getLine :: IO String
getLine = hGetLine stdin

getContents :: IO String
getContents = hGetContents stdin

interact :: (String -> String) -> IO ()
interact f = getContents >>= putStr . f

-- Errors
data IOError = IOError String
    deriving (Show, Eq)

instance Exception IOError

ioError :: IOError -> IO a
ioError = throw

userError :: String -> IOError
userError = IOError

catch :: Exception e => IO a -> (e -> IO a) -> IO a
catch = catchException

try :: Exception e => IO a -> IO (Either e a)
try action = catch (fmap Right action) (return . Left)

-- Internal
foreign import ccall "bhc_throw" throw :: Exception e => e -> a
foreign import ccall "bhc_catch" catchException :: Exception e => IO a -> (e -> IO a) -> IO a

class (Show e) => Exception e
