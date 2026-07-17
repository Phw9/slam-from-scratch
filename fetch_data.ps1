param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$DataDir = Join-Path $ScriptDir "image"
$ImageDir = Join-Path $DataDir "image_0"
$DataUrl = if ([string]::IsNullOrWhiteSpace($env:MVO_DATA_URL)) {
    "https://github.com/Phw9/slam-from-scratch/releases/download/kitti00-data/kitti00_image0.tar.gz"
} else {
    $env:MVO_DATA_URL
}

$HasImages = (Test-Path $ImageDir) -and
    (@(Get-ChildItem -Path $ImageDir -File -ErrorAction SilentlyContinue).Count -gt 0)

if ($Force -or -not $HasImages) {
    $Archive = Join-Path $DataDir "kitti00_image0.tar.gz"
    New-Item -ItemType Directory -Force $DataDir | Out-Null
    Write-Host "data=downloading url=$DataUrl"
    $CurlExe = Join-Path $env:SystemRoot "System32\curl.exe"
    if (Test-Path $CurlExe) {
        & $CurlExe -L --fail --retry 3 -o $Archive $DataUrl
        if ($LASTEXITCODE -ne 0) {
            throw "Image data download failed: $DataUrl"
        }
    } else {
        Invoke-WebRequest -Uri $DataUrl -OutFile $Archive
    }
    Write-Host "data=extracting archive=$Archive"
    $TarExe = Join-Path $env:SystemRoot "System32\tar.exe"
    if (!(Test-Path $TarExe)) {
        $TarExe = "tar"
    }
    & $TarExe -xzf $Archive -C $DataDir
    if ($LASTEXITCODE -ne 0) {
        throw "Image data extraction failed: $Archive"
    }
    Remove-Item $Archive -Force
    $Count = @(Get-ChildItem -Path $ImageDir -File).Count
    Write-Host "data=ready image_dir=$ImageDir files=$Count"
}
