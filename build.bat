@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
"%CMAKE%" -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
"%CMAKE%" --build build
