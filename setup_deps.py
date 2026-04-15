#!/usr/bin/env python
"""Install and verify dependencies for Quantum IDS project."""
import sys
import subprocess
import os

os.chdir(r"C:\Users\HPC\Desktop\Quantum_IDS\MINOR PROJECT")

print("="*70)
print("STEP 1: Detect Python and pip versions")
print("="*70)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

result = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
pip_version = result.stdout.strip()
print(f"Pip: {pip_version}\n")

print("="*70)
print("STEP 2: Read requirements and prepare clean format")
print("="*70)
# Read the requirements file and clean it
with open("qids_shap/requirements.txt", "r") as f:
    lines = f.readlines()

# Parse requirements (remove numbering if present)
requirements = []
for line in lines:
    line = line.strip()
    if line and not line.startswith("#"):
        # Remove leading numbers and dots (e.g., "1. " or "11.")
        if line[0].isdigit():
            # Find where the actual package name starts
            for i, char in enumerate(line):
                if char.isalpha():
                    line = line[i:]
                    break
        requirements.append(line)

if requirements:
    print(f"Found {len(requirements)} packages to install:")
    for req in requirements:
        print(f"  - {req}")
else:
    print("No requirements found!")
    sys.exit(1)

print("\n" + "="*70)
print("STEP 3: Install dependencies from requirements")
print("="*70)
result = subprocess.run(
    [sys.executable, "-m", "pip", "install"] + requirements,
    capture_output=True,
    text=True
)

# Print installation output (abbreviated for successful packages)
output_lines = result.stdout.split('\n')
for line in output_lines:
    if any(x in line for x in ['Successfully', 'Requirement already', 'ERROR', 'error', 'warning']):
        print(line)

if result.returncode != 0:
    print("\n⚠️  INSTALLATION HAD ERRORS:")
    print(result.stderr)
else:
    print("✓ Installation completed successfully")

print("\n" + "="*70)
print("STEP 4: Run pip check for conflicts")
print("="*70)
result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
check_output = result.stdout.strip()
print(check_output if check_output else "✓ No conflicts found - all dependencies are compatible!")

if result.returncode != 0:
    print("\n⚠️  Conflicts detected. Attempting to resolve...")
    print(result.stderr)
else:
    print("\n" + "="*70)
    print("✓ FINAL SUMMARY")
    print("="*70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Pip: {pip_version.split()[1]}")
    print(f"Packages installed: {len(requirements)}")
    print("Status: All dependencies compatible - pip check clean ✓")
