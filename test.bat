@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: Smart Document Assistant Test Runner for Windows
:: Uses uv for ultra-fast Python package management

echo 🚀 Smart Document Assistant Test Runner
echo.

:: Change to backend directory
cd /d "%~dp0backend"

:: Function to check if uv is installed
call :check_uv
if errorlevel 1 exit /b 1

:: Parse command
set "command=%~1"
if "%~1"=="" set "command=unit"

:: Handle commands
goto :command_%command% 2>nul || goto :command_help

:command_setup
    echo 📦 Setting up test environment...
    if not exist ".venv-test" (
        echo Creating test virtual environment...
        uv venv .venv-test
    )
    echo Installing dependencies with uv...
    uv pip install -e ".[test]" --python .venv-test
    echo ✅ Test environment ready!
    goto :eof

:command_unit
    call :setup_env
    echo 🧪 Running unit tests...
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "not slow"
    goto :eof

:command_all
    call :setup_env
    echo 🧪 Running all tests...
    uv run --python .venv-test pytest tests/unit -v --tb=short
    goto :eof

:command_coverage
    call :setup_env
    echo 📊 Running tests with coverage...
    uv run --python .venv-test pytest tests/unit -v --tb=short --cov=app --cov-report=term-missing --cov-report=html
    echo.
    echo ✅ Coverage report generated in htmlcov/index.html
    goto :eof

:command_document
    call :setup_env
    echo 🧪 Running document parsing tests...
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "document"
    goto :eof

:command_vector
    call :setup_env
    echo 🧪 Running vector operation tests...
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "vector"
    goto :eof

:command_embedding
    call :setup_env
    echo 🧪 Running embedding tests...
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "embedding"
    goto :eof

:command_storage
    call :setup_env
    echo 🧪 Running storage tests...
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "storage"
    goto :eof

:command_retrieval
    call :setup_env
    echo 🧪 Running retrieval tests...
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "retrieval"
    goto :eof

:command_prompt
    call :setup_env
    echo 🧪 Running prompt generation tests...
    uv run --python .venv-test pytest tests/unit -v --tb=short -m "prompt"
    goto :eof

:command_pattern
    if "%~2"=="" (
        echo ❌ Please provide a pattern
        echo Usage: test.bat pattern ^<test_pattern^>
        exit /b 1
    )
    call :setup_env
    echo 🧪 Running tests matching: %~2
    uv run --python .venv-test pytest tests/unit -v --tb=short -k "%~2"
    goto :eof

:command_clean
    echo 🧹 Cleaning test environment...
    if exist ".venv-test" rmdir /s /q .venv-test
    if exist "htmlcov" rmdir /s /q htmlcov
    if exist ".pytest_cache" rmdir /s /q .pytest_cache
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
    echo ✅ Cleaned!
    goto :eof

:command_help
    echo Usage: test.bat [command]
    echo.
    echo Commands:
    echo   setup           Setup test environment
    echo   unit            Run unit tests ^(default, excludes slow tests^)
    echo   all             Run all tests including slow tests
    echo   coverage        Run tests with coverage report
    echo   document        Run document parsing tests
    echo   vector          Run vector operation tests
    echo   embedding       Run embedding tests
    echo   storage         Run storage tests
    echo   retrieval       Run retrieval tests
    echo   prompt          Run prompt generation tests
    echo   pattern ^<p^>    Run tests matching pattern
    echo   clean           Clean test environment and cache
    echo   help            Show this help message
    echo.
    echo Examples:
    echo   test.bat                  Run unit tests
    echo   test.bat coverage         Run with coverage
    echo   test.bat pattern split    Run tests with 'split' in name
    echo   test.bat document         Run document-related tests
    goto :eof

:: Helper functions
check_uv
    uv --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ uv is not installed
        echo Install uv: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        exit /b 1
    )
    exit /b 0

setup_env
    if not exist ".venv-test" (
        echo 📦 Setting up test environment...
        uv venv .venv-test
        uv pip install -e ".[test]" --python .venv-test
    )
    exit /b 0

:eof
    echo.
    echo ✅ Done!
    endlocal
