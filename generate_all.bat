@echo off
REM Batch script to generate data for all categories sequentially

echo ================================================================================
echo Mental Health Data Generator - Batch Processing
echo ================================================================================
echo.

set COUNT=%1
if "%COUNT%"=="" set COUNT=10

echo Generating %COUNT% sentences per category...
echo.

echo [1/4] Generating Anxiety data...
python main.py --category "Anxiety" --count %COUNT%
if errorlevel 1 (
    echo ERROR: Anxiety generation failed!
    exit /b 1
)
echo.

echo [2/4] Generating Depression data...
python main.py --category "Depression" --count %COUNT%
if errorlevel 1 (
    echo ERROR: Depression generation failed!
    exit /b 1
)
echo.

echo [3/4] Generating Psychosis data...
python main.py --category "Psychosis" --count %COUNT%
if errorlevel 1 (
    echo ERROR: Psychosis generation failed!
    exit /b 1
)
echo.

echo [4/4] Generating Mania data...
python main.py --category "Mania" --count %COUNT%
if errorlevel 1 (
    echo ERROR: Mania generation failed!
    exit /b 1
)
echo.

echo ================================================================================
echo SUCCESS! Generated %COUNT% sentences for each of 4 categories
echo.
echo Output files created in: data\output\
echo   - generated_sentences_v*_Anxiety.json
echo   - generated_sentences_v*_Depression.json
echo   - generated_sentences_v*_Psychosis.json
echo   - generated_sentences_v*_Mania.json
echo.
echo Each file is automatically versioned to prevent overwrites!
echo ================================================================================
