-- |
-- Module      : BHC.Data.Time
-- Description : Date and time handling
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Time representation and manipulation.

module BHC.Data.Time (
    -- * Duration
    Duration,
    fromNanos, fromMicros, fromMillis, fromSecs, fromMins, fromHours, fromDays,
    toNanos, toMicros, toMillis, toSecs,
    
    -- * Instant
    Instant,
    now,
    elapsed,
    
    -- * Date
    Date,
    date,
    year, month, day,
    dayOfWeek, dayOfYear, weekOfYear,
    isLeapYear,
    
    -- * Time
    Time,
    time,
    hour, minute, second, nanosecond,
    
    -- * DateTime
    DateTime,
    dateTime,
    getDate, getTime,
    
    -- * Parsing and formatting
    parseDateTime, formatDateTime,
    parseDate, formatDate,
    
    -- * Utilities
    measure,
    sleep,
) where

import BHC.Prelude

-- | A duration of time in nanoseconds.
newtype Duration = Duration { durationNanos :: Integer }
    deriving (Eq, Ord, Show, Read)

instance Semigroup Duration where
    Duration a <> Duration b = Duration (a + b)

instance Monoid Duration where
    mempty = Duration 0

instance Num Duration where
    Duration a + Duration b = Duration (a + b)
    Duration a - Duration b = Duration (a - b)
    Duration a * Duration b = Duration (a * b)
    negate (Duration a) = Duration (negate a)
    abs (Duration a) = Duration (abs a)
    signum (Duration a) = Duration (signum a)
    fromInteger = Duration

-- Duration constructors
fromNanos :: Integer -> Duration
fromNanos = Duration

fromMicros :: Integer -> Duration
fromMicros us = Duration (us * 1000)

fromMillis :: Integer -> Duration
fromMillis ms = Duration (ms * 1000000)

fromSecs :: Integer -> Duration
fromSecs s = Duration (s * 1000000000)

fromMins :: Integer -> Duration
fromMins m = fromSecs (m * 60)

fromHours :: Integer -> Duration
fromHours h = fromMins (h * 60)

fromDays :: Integer -> Duration
fromDays d = fromHours (d * 24)

-- Duration accessors
toNanos :: Duration -> Integer
toNanos = durationNanos

toMicros :: Duration -> Integer
toMicros (Duration n) = n `div` 1000

toMillis :: Duration -> Integer
toMillis (Duration n) = n `div` 1000000

toSecs :: Duration -> Integer
toSecs (Duration n) = n `div` 1000000000

-- | A point in time.
data Instant = Instant
    { instantSecs :: !Integer
    , instantNanos :: !Int
    }
    deriving (Eq, Ord, Show, Read)

-- | Get the current instant.
foreign import ccall "bhc_instant_now" now :: IO Instant

-- | Time elapsed since an instant.
elapsed :: Instant -> IO Duration
elapsed start = do
    end <- now
    return $ Duration $
        (instantSecs end - instantSecs start) * 1000000000 +
        fromIntegral (instantNanos end - instantNanos start)

-- | A calendar date.
data Date = Date
    { dateYear :: !Int
    , dateMonth :: !Int
    , dateDay :: !Int
    }
    deriving (Eq, Ord, Show, Read)

-- | Create a date.
date :: Int -> Int -> Int -> Date
date y m d
    | m < 1 || m > 12 = error "invalid month"
    | d < 1 || d > daysInMonth y m = error "invalid day"
    | otherwise = Date y m d

year :: Date -> Int
year = dateYear

month :: Date -> Int
month = dateMonth

day :: Date -> Int
day = dateDay

dayOfWeek :: Date -> Int
dayOfWeek (Date y m d) =
    let a = (14 - m) `div` 12
        y' = y - a
        m' = m + 12 * a - 2
    in (d + y' + y' `div` 4 - y' `div` 100 + y' `div` 400 + (31 * m') `div` 12) `mod` 7

dayOfYear :: Date -> Int
dayOfYear (Date y m d) = sum [daysInMonth y i | i <- [1..m-1]] + d

weekOfYear :: Date -> Int
weekOfYear dt = (dayOfYear dt + 6) `div` 7

isLeapYear :: Int -> Bool
isLeapYear y = y `mod` 4 == 0 && (y `mod` 100 /= 0 || y `mod` 400 == 0)

daysInMonth :: Int -> Int -> Int
daysInMonth y m
    | m `elem` [1, 3, 5, 7, 8, 10, 12] = 31
    | m `elem` [4, 6, 9, 11] = 30
    | m == 2 && isLeapYear y = 29
    | m == 2 = 28
    | otherwise = error "invalid month"

-- | A time of day.
data Time = Time
    { timeHour :: !Int
    , timeMinute :: !Int
    , timeSecond :: !Int
    , timeNano :: !Int
    }
    deriving (Eq, Ord, Show, Read)

-- | Create a time.
time :: Int -> Int -> Int -> Time
time h m s
    | h < 0 || h > 23 = error "invalid hour"
    | m < 0 || m > 59 = error "invalid minute"
    | s < 0 || s > 59 = error "invalid second"
    | otherwise = Time h m s 0

hour :: Time -> Int
hour = timeHour

minute :: Time -> Int
minute = timeMinute

second :: Time -> Int
second = timeSecond

nanosecond :: Time -> Int
nanosecond = timeNano

-- | A date and time.
data DateTime = DateTime
    { dtDate :: !Date
    , dtTime :: !Time
    }
    deriving (Eq, Ord, Show, Read)

-- | Create a datetime.
dateTime :: Date -> Time -> DateTime
dateTime = DateTime

getDate :: DateTime -> Date
getDate = dtDate

getTime :: DateTime -> Time
getTime = dtTime

-- | Parse an ISO 8601 datetime.
parseDateTime :: String -> Maybe DateTime
parseDateTime s = case break (== 'T') s of
    (dateStr, 'T':timeStr) -> do
        d <- parseDate dateStr
        t <- parseTime timeStr
        return (DateTime d t)
    _ -> Nothing

-- | Format as ISO 8601.
formatDateTime :: DateTime -> String
formatDateTime (DateTime d t) = formatDate d ++ "T" ++ formatTime t

-- | Parse a date (YYYY-MM-DD).
parseDate :: String -> Maybe Date
parseDate s = case break (== '-') s of
    (yStr, '-':rest) -> case break (== '-') rest of
        (mStr, '-':dStr) -> do
            y <- readMaybe yStr
            m <- readMaybe mStr
            d <- readMaybe dStr
            if m >= 1 && m <= 12 && d >= 1 && d <= daysInMonth y m
                then Just (Date y m d)
                else Nothing
        _ -> Nothing
    _ -> Nothing

-- | Format a date as YYYY-MM-DD.
formatDate :: Date -> String
formatDate (Date y m d) = 
    show y ++ "-" ++ pad2 m ++ "-" ++ pad2 d

-- | Parse a time (HH:MM:SS).
parseTime :: String -> Maybe Time
parseTime s = case break (== ':') s of
    (hStr, ':':rest) -> case break (== ':') rest of
        (mStr, ':':sStr) -> do
            h <- readMaybe hStr
            m <- readMaybe mStr
            sec <- readMaybe (takeWhile (\c -> c >= '0' && c <= '9') sStr)
            if h >= 0 && h <= 23 && m >= 0 && m <= 59 && sec >= 0 && sec <= 59
                then Just (Time h m sec 0)
                else Nothing
        (mStr, "") -> do
            h <- readMaybe hStr
            m <- readMaybe mStr
            if h >= 0 && h <= 23 && m >= 0 && m <= 59
                then Just (Time h m 0 0)
                else Nothing
        _ -> Nothing
    _ -> Nothing

-- | Format a time as HH:MM:SS.
formatTime :: Time -> String
formatTime (Time h m s _) =
    pad2 h ++ ":" ++ pad2 m ++ ":" ++ pad2 s

pad2 :: Int -> String
pad2 n = if n < 10 then '0' : show n else show n

readMaybe :: Read a => String -> Maybe a
readMaybe s = case reads s of
    [(x, "")] -> Just x
    _ -> Nothing

-- | Measure the time taken by an action.
measure :: IO a -> IO (a, Duration)
measure action = do
    start <- now
    result <- action
    duration <- elapsed start
    return (result, duration)

-- | Sleep for a duration.
foreign import ccall "bhc_sleep" sleep :: Duration -> IO ()
