@echo off
REM Installation script for Quantum IDS dependencies
cd /d "C:\Users\HPC\Desktop\Quantum_IDS\MINOR PROJECT"

echo.
echo ========================================================================
echo STEP 1: Detect Python and pip versions
echo ========================================================================
python --version
for /f "tokens=*" %%i in ('python -m pip --version') do echo %%i

echo.
echo ========================================================================
echo STEP 2: Install dependencies from qids_shap\requirements.txt
echo ========================================================================
REM Parse requirements and install
python -m pip install shap scikit-learn xgboost joblib pandas numpy pyarrow matplotlib streamlit lime

echo.
echo ========================================================================
echo STEP 3: Run pip check for conflicts
echo ========================================================================
python -m pip check

echo.
echo Done!
pause
