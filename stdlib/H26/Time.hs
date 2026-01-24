-- |
-- Module      : H26.Time
-- Description : Time and duration types
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Time measurement, durations, and calendar operations.
--
-- = Overview
--
-- This module provides comprehensive time handling:
--
-- * 'Duration' — Time spans with nanosecond precision
-- * 'Instant' — Monotonic timestamps for measuring elapsed time
-- * 'UTCTime' — Wall-clock time (affected by system adjustments)
-- * Calendar types for dates and times
--
-- = Quick Start
--
-- @
-- import H26.Time
--
-- -- Measure execution time
-- main :: IO ()
-- main = do
--     start <- now
--     result <- performComputation
--     elapsed <- elapsed start
--     putStrLn $ \"Took \" ++ show (toMilliseconds elapsed) ++ \"ms\"
--
-- -- Create dates and times
-- let day = fromGregorian 2026 1 24
--     (y, m, d) = toGregorian day  -- (2026, 1, 24)
--
-- -- Durations
-- let timeout = seconds 30
--     delay = milliseconds 100
--     total = addDuration timeout delay
-- @
--
-- = Instant vs UTCTime
--
-- Use 'Instant' for performance measurement and timeouts:
--
-- @
-- start <- now          -- Monotonic, never goes backwards
-- -- ... computation ...
-- elapsed <- elapsed start
-- @
--
-- Use 'UTCTime' for wall-clock time and calendars:
--
-- @
-- utc <- getCurrentTime
-- let formatted = formatTime iso8601Format utc
-- @
--
-- = Timeout and Delay
--
-- @
-- -- Run with timeout
-- result <- timeout (seconds 5) longOperation
-- case result of
--     Just x  -> use x
--     Nothing -> handleTimeout
--
-- -- Delay execution
-- delay (milliseconds 100)
-- @
--
-- = See Also
--
-- * "BHC.Data.Time" for the underlying implementation
-- * "H26.Concurrency" for deadline-aware concurrency

{-# HASKELL_EDITION 2026 #-}

module H26.Time
  ( -- * Duration
    Duration
  , Nanoseconds
  , Microseconds
  , Milliseconds
  , Seconds

    -- * Duration Construction
  , nanoseconds
  , microseconds
  , milliseconds
  , seconds
  , minutes
  , hours
  , days

    -- * Duration Queries
  , toNanoseconds
  , toMicroseconds
  , toMilliseconds
  , toSeconds
  , toMinutes
  , toHours
  , toDays

    -- * Duration Arithmetic
  , addDuration
  , subtractDuration
  , multiplyDuration
  , divideDuration

    -- * Instant (Monotonic Time)
  , Instant
  , now
  , elapsed
  , durationSince
  , durationBetween

    -- * UTCTime
  , UTCTime
  , getCurrentTime
  , utcToInstant
  , diffUTCTime
  , addUTCTime

    -- * LocalTime
  , LocalTime
  , TimeZone
  , utcToLocalTime
  , localTimeToUTC
  , getCurrentTimeZone
  , getTimeZone

    -- * Calendar Types
  , Day
  , Year
  , Month
  , DayOfMonth
  , DayOfWeek(..)

    -- * Date Construction
  , fromGregorian
  , toGregorian
  , fromOrdinalDate
  , toOrdinalDate

    -- * Date Queries
  , dayOfWeek
  , isLeapYear
  , daysInMonth

    -- * Date Arithmetic
  , addDays
  , diffDays
  , addMonths
  , addYears

    -- * TimeOfDay
  , TimeOfDay
  , midnight
  , midday
  , makeTimeOfDay
  , timeOfDayToTime
  , timeToTimeOfDay
  , todHour
  , todMin
  , todSec

    -- * Formatting
  , formatTime
  , parseTimeM
  , defaultTimeLocale
  , iso8601Format

    -- * Timing Operations
  , timeout
  , delay
  , measure
  , measureIO

    -- * Clock Types
  , MonotonicClock
  , SystemClock
  , getMonotonicTime
  , getSystemTime
  ) where

-- | Duration represents a span of time.
--
-- Internally stored as nanoseconds for precision.
-- Can represent durations up to ~292 years.
newtype Duration = Duration Int64

-- | Type aliases for clarity.
type Nanoseconds = Int64
type Microseconds = Int64
type Milliseconds = Int64
type Seconds = Double

-- | Monotonic timestamp for measuring elapsed time.
--
-- Unlike system time, this never goes backwards and is not
-- affected by system clock adjustments. Use for performance
-- measurement and timeouts.
data Instant

-- | UTC timestamp.
data UTCTime

-- | Local time (without timezone).
data LocalTime

-- | Timezone offset and name.
data TimeZone

-- | Calendar day.
newtype Day = Day Int32

-- | Calendar year.
type Year = Int

-- | Calendar month (1-12).
type Month = Int

-- | Day of month (1-31).
type DayOfMonth = Int

-- | Day of week.
data DayOfWeek
  = Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
  deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Time of day (hours, minutes, seconds).
data TimeOfDay = TimeOfDay
  { todHour :: !Int
  , todMin  :: !Int
  , todSec  :: !Double
  }

-- | Create duration from nanoseconds.
nanoseconds :: Int64 -> Duration

-- | Create duration from microseconds.
microseconds :: Int64 -> Duration

-- | Create duration from milliseconds.
milliseconds :: Int64 -> Duration

-- | Create duration from seconds.
seconds :: Double -> Duration

-- | Create duration from minutes.
minutes :: Double -> Duration

-- | Create duration from hours.
hours :: Double -> Duration

-- | Create duration from days.
days :: Double -> Duration

-- | Convert to nanoseconds.
toNanoseconds :: Duration -> Int64

-- | Convert to microseconds.
toMicroseconds :: Duration -> Int64

-- | Convert to milliseconds.
toMilliseconds :: Duration -> Int64

-- | Convert to fractional seconds.
toSeconds :: Duration -> Double

-- | Convert to fractional minutes.
toMinutes :: Duration -> Double

-- | Convert to fractional hours.
toHours :: Duration -> Double

-- | Convert to fractional days.
toDays :: Duration -> Double

-- | Add two durations.
addDuration :: Duration -> Duration -> Duration

-- | Subtract durations.
subtractDuration :: Duration -> Duration -> Duration

-- | Multiply duration by scalar.
multiplyDuration :: Duration -> Double -> Duration

-- | Divide duration by scalar.
divideDuration :: Duration -> Double -> Duration

-- | Get current monotonic instant.
--
-- This is the primary function for measuring elapsed time.
now :: IO Instant

-- | Get elapsed time since an instant.
elapsed :: Instant -> IO Duration

-- | Duration since a past instant.
durationSince :: Instant -> Instant -> Duration

-- | Duration between two instants.
durationBetween :: Instant -> Instant -> Duration

-- | Get current UTC time.
getCurrentTime :: IO UTCTime

-- | Convert UTC time to monotonic instant (approximate).
utcToInstant :: UTCTime -> IO Instant

-- | Difference between UTC times.
diffUTCTime :: UTCTime -> UTCTime -> Duration

-- | Add duration to UTC time.
addUTCTime :: Duration -> UTCTime -> UTCTime

-- | Convert UTC to local time.
utcToLocalTime :: TimeZone -> UTCTime -> LocalTime

-- | Convert local time to UTC.
localTimeToUTC :: TimeZone -> LocalTime -> UTCTime

-- | Get current timezone.
getCurrentTimeZone :: IO TimeZone

-- | Get timezone for a specific time.
getTimeZone :: UTCTime -> IO TimeZone

-- | Create a day from year, month, day.
fromGregorian :: Year -> Month -> DayOfMonth -> Day

-- | Extract year, month, day from a Day.
toGregorian :: Day -> (Year, Month, DayOfMonth)

-- | Create from ordinal date (year, day-of-year).
fromOrdinalDate :: Year -> Int -> Day

-- | Convert to ordinal date.
toOrdinalDate :: Day -> (Year, Int)

-- | Get day of week.
dayOfWeek :: Day -> DayOfWeek

-- | Check if year is a leap year.
isLeapYear :: Year -> Bool

-- | Number of days in a month.
daysInMonth :: Year -> Month -> Int

-- | Add days to a date.
addDays :: Int -> Day -> Day

-- | Difference in days.
diffDays :: Day -> Day -> Int

-- | Add months to a date.
addMonths :: Int -> Day -> Day

-- | Add years to a date.
addYears :: Int -> Day -> Day

-- | Midnight (00:00:00).
midnight :: TimeOfDay

-- | Midday (12:00:00).
midday :: TimeOfDay

-- | Create time of day.
makeTimeOfDay :: Int -> Int -> Double -> TimeOfDay

-- | Convert TimeOfDay to duration since midnight.
timeOfDayToTime :: TimeOfDay -> Duration

-- | Convert duration since midnight to TimeOfDay.
timeToTimeOfDay :: Duration -> TimeOfDay

-- | Format time using strftime-style format string.
formatTime :: String -> UTCTime -> String

-- | Parse time from string.
parseTimeM :: Monad m => Bool -> String -> String -> m UTCTime

-- | Default time locale for formatting.
defaultTimeLocale :: TimeLocale

-- | ISO 8601 format string.
iso8601Format :: String

-- | Run action with timeout.
--
-- Returns Nothing if timeout expires before action completes.
timeout :: Duration -> IO a -> IO (Maybe a)

-- | Delay for specified duration.
delay :: Duration -> IO ()

-- | Measure duration of pure computation.
measure :: a -> (a, Duration)

-- | Measure duration of IO action.
measureIO :: IO a -> IO (a, Duration)

-- | Monotonic clock source.
data MonotonicClock

-- | System clock source.
data SystemClock

-- | Read monotonic clock directly.
getMonotonicTime :: IO Instant

-- | Read system clock directly.
getSystemTime :: IO UTCTime

-- Internal types
data TimeLocale

-- This is a specification file.
-- Actual implementation provided by the compiler.
