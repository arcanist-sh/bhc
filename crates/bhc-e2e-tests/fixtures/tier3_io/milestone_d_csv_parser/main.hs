main = do
    let input = "name,age,city\nAlice,30,Zurich\nBob,25,Basel\n"
    result <- evalStateT parseCSV input
    putStrLn result

parseCSV = do
    header <- parseLine
    rows <- parseRows
    return (formatTable header rows)

parseRows = do
    s <- get
    case s of
        "" -> return []
        _  -> do
            row <- parseLine
            rest <- parseRows
            return (row : rest)

parseLine = do
    s <- get
    case s of
        "" -> return []
        _  -> do
            field <- parseField
            rest <- parseRestOfLine
            return (field : rest)

parseRestOfLine = do
    s <- get
    case s of
        "" -> return []
        _  -> case head s of
            '\n' -> do
                put (tail s)
                return []
            ','  -> do
                put (tail s)
                field <- parseField
                rest <- parseRestOfLine
                return (field : rest)
            _    -> return []

parseField = do
    s <- get
    put (dropField s)
    return (takeField s)

takeField s = case s of
    "" -> ""
    _  -> case head s of
        ',' -> ""
        '\n' -> ""
        _   -> head s : takeField (tail s)

dropField s = case s of
    "" -> ""
    _  -> case head s of
        ',' -> s
        '\n' -> s
        _   -> dropField (tail s)

formatTable header rows =
    formatRow header ++ "\n" ++ formatSep header ++ "\n" ++ formatRows rows

formatRows rows = case rows of
    [] -> ""
    _  -> formatRow (head rows) ++ "\n" ++ formatRows (tail rows)

formatRow fields = case fields of
    [] -> ""
    _  -> case tail fields of
        [] -> head fields
        _  -> head fields ++ " | " ++ formatRow (tail fields)

formatSep fields = case fields of
    [] -> ""
    _  -> case tail fields of
        [] -> makeDashes (length (head fields))
        _  -> makeDashes (length (head fields)) ++ "-+-" ++ formatSep (tail fields)

makeDashes n = case n of
    0 -> ""
    _ -> "-" ++ makeDashes (n - 1)
