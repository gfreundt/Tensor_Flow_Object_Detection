@echo off

echo Step 0
python install.py 0
if %errorlevel% neq 0 exit

echo Step 1
.\TFODCourse\tfod\Scripts\python.exe install.py 1
if %errorlevel% neq 0 exit

echo Step 1.5
.\TFODCourse\tfod\Scripts\pyrcc5 -o .\TFODCourse\Tensorflow\labelImg\libs\resources.py .\TFODCourse\Tensorflow\labelImg\resources.qrc
.\TFODCourse\tfod\Scripts\python.exe .\TFODCourse\Tensorflow\labelImg\labelImg.py
if %errorlevel% neq 0 exit

echo Step 2
.\TFODCourse\tfod\Scripts\python.exe install.py 2
if %errorlevel% neq 0 exit

echo Step 3
.\TFODCourse\tfod\Scripts\python.exe install.py 3