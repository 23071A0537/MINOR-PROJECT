param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$logPath = Join-Path $ProjectRoot "artifacts\reports\post_v7_hybrid_autorun.log"
$logDir = Split-Path $logPath -Parent
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

"[$(Get-Date -Format s)] watcher started" | Out-File -FilePath $logPath -Encoding utf8

while ($true) {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -match "vqc_v7_phase1_train_complete.py"
    }

    if (-not $procs) {
        break
    }

    "[$(Get-Date -Format s)] trainer active (count=$($procs.Count)); waiting 30s" | Out-File -FilePath $logPath -Append -Encoding utf8
    Start-Sleep -Seconds 30
}

"[$(Get-Date -Format s)] trainer finished; running post_v7_hybrid_sweep.py" | Out-File -FilePath $logPath -Append -Encoding utf8

$pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $ProjectRoot "scripts\post_v7_hybrid_sweep.py"

& $pythonPath $scriptPath *>> $logPath
$exitCode = $LASTEXITCODE

"[$(Get-Date -Format s)] post-sweep exit_code=$exitCode" | Out-File -FilePath $logPath -Append -Encoding utf8
exit $exitCode
