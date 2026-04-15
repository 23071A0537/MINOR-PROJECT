@echo off
setlocal

cd /d "C:\Users\HPC\Desktop\Quantum_IDS\MINOR PROJECT"

echo ================================================================
echo RTX A2000 GPU setup for Quantum IDS (Windows)
echo ================================================================

echo.
echo [1/5] Python and pip
python --version
python -m pip --version

echo.
echo [2/5] Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo.
echo [3/5] Install project dependencies
python -m pip install -r qids_shap\requirements.txt
if errorlevel 1 goto :error

echo.
echo [4/5] Install CUDA-enabled PyTorch (cu121 wheels)
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :error

echo.
echo [5/5] Verify GPU stack
python scripts\verify_gpu_stack.py
if errorlevel 1 goto :error

echo.
echo ================================================================
echo Setup complete.
echo Use this training entrypoint for GPU on Windows:
echo   python VQC\vqc_v7_phase1_train_pytorch.py
echo ================================================================
goto :end

:error
echo.
echo Setup failed. Fix the error shown above and run again.
exit /b 1

:end
endlocal
