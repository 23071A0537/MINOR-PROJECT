#!/usr/bin/env python
import sys
import subprocess

# Show Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Show pip version
result = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
print(f"Pip version: {result.stdout.strip()}")

# Install requirements
print("\n" + "="*60)
print("Installing dependencies from qids_shap/requirements.txt...")
print("="*60)
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "qids_shap/requirements.txt"],
    cwd=r"C:\Users\HPC\Desktop\Quantum_IDS\MINOR PROJECT",
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# Run pip check
print("\n" + "="*60)
print("Running pip check...")
print("="*60)
result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
print(result.stdout if result.stdout else "All dependencies OK!")
if result.returncode != 0:
    print("CONFLICTS DETECTED:", result.stderr)
