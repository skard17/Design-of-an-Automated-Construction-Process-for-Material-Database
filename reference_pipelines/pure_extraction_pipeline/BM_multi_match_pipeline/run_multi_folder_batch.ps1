$ErrorActionPreference = "Stop"

$root = "C:\Users\Administrator\Desktop\fsdownload\sc-ie"
$pipelineRoot = Join-Path $root "BM_multi_match_pipeline"
$sourceDir = Join-Path $root "multi"
$papersDir = Join-Path $pipelineRoot "papers"
$logDir = Join-Path $pipelineRoot "logs"
$logPath = Join-Path $logDir "run_multi_folder_batch.log"
$keyFile = Join-Path $root ".siliconflow_keys.local.sh"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $logPath -Value $line
    Write-Output $line
}

function Invoke-LoggedPython {
    param(
        [string]$ScriptPath,
        [string]$PaperId
    )

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python $ScriptPath --paper_id $PaperId *>> $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "python exit code $LASTEXITCODE for $ScriptPath --paper_id $PaperId"
        }
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}

$content = Get-Content -Raw -Path $keyFile
$match = [regex]::Match($content, 'sk-[A-Za-z0-9]+')
if (-not $match.Success) {
    throw "API key not found in $keyFile"
}

$env:SILICONFLOW_API_KEY = $match.Value
$env:SILICONFLOW_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"
$env:SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
$env:BM_MULTI_TIMEOUT_SEC = "900"

Get-ChildItem $sourceDir -Filter *.md | Copy-Item -Destination $papersDir -Force

$paperFiles = Get-ChildItem $sourceDir -Filter *.md | Sort-Object Name

foreach ($paper in $paperFiles) {
    $paperId = $paper.BaseName
    $finalDir = Join-Path $pipelineRoot ("outputs\\multi_paper_final\\" + $paperId)
    $summaryPath = Join-Path $finalDir "run_summary.json"

    if (Test-Path $summaryPath) {
        Write-Log "skip $paperId (run_summary exists)"
        continue
    }

    Write-Log "start $paperId"
    try {
        Invoke-LoggedPython -ScriptPath (Join-Path $pipelineRoot "legacy_single_clone\\run_ie0_multi.py") -PaperId $paperId
        Invoke-LoggedPython -ScriptPath (Join-Path $pipelineRoot "legacy_single_clone\\run_fig_classify_multi.py") -PaperId $paperId
        Invoke-LoggedPython -ScriptPath (Join-Path $pipelineRoot "legacy_single_clone\\run_multi_sections_paper.py") -PaperId $paperId
        Write-Log "done $paperId"
    }
    catch {
        Write-Log ("error " + $paperId + " :: " + $_.Exception.Message)
    }
}

Write-Log "batch finished"
