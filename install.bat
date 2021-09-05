@echo off

echo Step 0
REM python install.py 0
if %errorlevel% neq 0 exit

echo Step 1
REM .\TFODCourse\tfod\Scripts\python.exe install.py 1
if %errorlevel% neq 0 exit

echo Step 1.5
REM .\TFODCourse\tfod\Scripts\pyrcc5 -o .\TFODCourse\Tensorflow\labelImg\libs\resources.py .\TFODCourse\Tensorflow\labelImg\resources.qrc
REM .\TFODCourse\tfod\Scripts\python.exe .\TFODCourse\Tensorflow\labelImg\labelImg.py
if %errorlevel% neq 0 exit

echo Step 2
REM .\TFODCourse\tfod\Scripts\python.exe install.py 2
if %errorlevel% neq 0 exit

echo Step 3
REM .\TFODCourse\tfod\Scripts\python.exe install.py 3

echo Step 4
REM .\TFODCourse\tfod\Scripts\python.exe install.py 4

echo Step 5
.\TFODCourse\tfod\Scripts\python.exe install.py 5