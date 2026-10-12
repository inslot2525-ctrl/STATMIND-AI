# Starts the whole local stack, each service in its own window:
#   lakeFS   http://127.0.0.1:8001   data versioning   (only if .tools\lakefs\lakefs.exe exists)
#   MLflow   http://127.0.0.1:5000   experiment tracking and model registry
#   API      http://127.0.0.1:8000   StatMind backend
#   App      http://localhost:5173   StatMind frontend
#
# Usage:  .\start-dev.ps1            everything (services already running are left alone)
#         .\start-dev.ps1 -Plain     just the API and the app, no lakeFS or MLflow
#
# lakeFS credentials are generated on first run and kept in lakefs-data\credentials.json
# (git-ignored). This is a single-user local setup; do not expose it to a network.
param([switch]$Plain)

$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$mlflow = Join-Path $root ".venv\Scripts\mlflow.exe"
$lakefs = Join-Path $root ".tools\lakefs\lakefs.exe"

function Start-Service-Window($title, $workdir, $command) {
    Start-Process powershell -WorkingDirectory $workdir -ArgumentList @(
        "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle = '$title'; $command"
    )
}

function Test-Up($url) {
    try {
        Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Start a service unless it is already answering, so the script can be re-run
# to bring back only what was closed.
function Start-Unless-Up($title, $url, $workdir, $command) {
    if (Test-Up $url) {
        Write-Host "$title is already running"
        return
    }
    Start-Service-Window $title $workdir $command
    Wait-For $title $url
}

function Wait-For($name, $url) {
    foreach ($attempt in 1..60) {
        try {
            Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
            Write-Host "$name is up"
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    Write-Warning "$name did not answer at $url"
}

$backendEnv = ""

if (-not $Plain) {
    if (Test-Path $lakefs) {
        $data = Join-Path $root "lakefs-data"
        New-Item -ItemType Directory -Force $data | Out-Null
        # First run: make up credentials for this machine. Later runs reuse them.
        $credentialsFile = Join-Path $data "credentials.json"
        if (-not (Test-Path $credentialsFile)) {
            $alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
            $random = { param($length) -join (1..$length | ForEach-Object { $alphabet[(Get-Random -Maximum $alphabet.Length)] }) }
            @{
                mode              = "local"
                access_key_id     = "AKIA" + (& $random 16)
                secret_access_key = (& $random 40)
            } | ConvertTo-Json | Out-File -Encoding utf8 $credentialsFile
        }
        $credentials = Get-Content $credentialsFile -Raw | ConvertFrom-Json
        $mode = if ($credentials.mode -eq "quickstart") { "--quickstart" } else { "--local-settings" }

        $lakefsEnv = @(
            "`$env:LAKEFS_LISTEN_ADDRESS = '127.0.0.1:8001'"
            "`$env:LAKEFS_DATABASE_LOCAL_PATH = '$data\metadata'"
            "`$env:LAKEFS_BLOCKSTORE_LOCAL_PATH = '$data\blocks'"
            "`$env:LAKEFS_COMMITTED_LOCAL_CACHE_DIR = '$data\cache'"
            "`$env:LAKEFS_STATS_ENABLED = 'false'"
        )
        # Quickstart mode has fixed credentials and refuses to have them set
        if ($mode -eq "--local-settings") {
            $lakefsEnv += "`$env:LAKEFS_INSTALLATION_USER_NAME = 'admin'"
            $lakefsEnv += "`$env:LAKEFS_INSTALLATION_ACCESS_KEY_ID = '$($credentials.access_key_id)'"
            $lakefsEnv += "`$env:LAKEFS_INSTALLATION_SECRET_ACCESS_KEY = '$($credentials.secret_access_key)'"
        }
        $lakefsEnv = $lakefsEnv -join "; "
        Start-Unless-Up "lakeFS" "http://127.0.0.1:8001/api/v1/healthcheck" $root "$lakefsEnv; & '$lakefs' run $mode"

        $backendEnv += "`$env:LAKEFS_ENDPOINT = 'http://127.0.0.1:8001'; "
        $backendEnv += "`$env:LAKEFS_ACCESS_KEY_ID = '$($credentials.access_key_id)'; "
        $backendEnv += "`$env:LAKEFS_SECRET_ACCESS_KEY = '$($credentials.secret_access_key)'; "
    } else {
        Write-Warning "lakeFS is not installed at $lakefs, so files will be stored locally without versioning."
    }

    $mlflowData = Join-Path $root "mlflow-data"
    New-Item -ItemType Directory -Force $mlflowData | Out-Null
    Start-Unless-Up "MLflow" "http://127.0.0.1:5000/health" $mlflowData "& '$mlflow' server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts"
    $backendEnv += "`$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'; "
}

Start-Unless-Up "StatMind API" "http://127.0.0.1:8000/api/health" (Join-Path $root "backend") "$backendEnv& '$python' -m uvicorn main:app --port 8000"

Start-Unless-Up "StatMind app" "http://localhost:5173/" (Join-Path $root "frontend") "npm run dev -- --port 5173 --strictPort"

Write-Host ""
Write-Host "App      http://localhost:5173"
if (-not $Plain) {
    Write-Host "MLflow   http://127.0.0.1:5000"
    if (Test-Path $lakefs) {
        Write-Host "lakeFS   http://127.0.0.1:8001   (log in with the keys in lakefs-data\credentials.json)"
    }
}
Write-Host "Close the service windows to stop everything."
