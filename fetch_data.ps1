param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$DataDir = Join-Path $ScriptDir "image"
$ReleaseBase = "https://github.com/Phw9/slam-from-scratch/releases/download/kitti00-data"

function Get-Archive {
    param(
        [string]$ImageDir,
        [string]$ArchiveName,
        [string]$DataUrl
    )

    $HasImages = (Test-Path $ImageDir) -and
        (@(Get-ChildItem -Path $ImageDir -File -ErrorAction SilentlyContinue).Count -gt 0)

    if (-not $Force -and $HasImages) {
        return
    }

    $Archive = Join-Path $DataDir $ArchiveName
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

$LeftUrl = if ([string]::IsNullOrWhiteSpace($env:MVO_DATA_URL)) {
    "$ReleaseBase/kitti00_image0.tar.gz"
} else {
    $env:MVO_DATA_URL
}
$RightUrl = if ([string]::IsNullOrWhiteSpace($env:MVO_STEREO_DATA_URL)) {
    "$ReleaseBase/kitti00_image1.tar.gz"
} else {
    $env:MVO_STEREO_DATA_URL
}

Get-Archive -ImageDir (Join-Path $DataDir "image_0") `
    -ArchiveName "kitti00_image0.tar.gz" -DataUrl $LeftUrl
Get-Archive -ImageDir (Join-Path $DataDir "image_1") `
    -ArchiveName "kitti00_image1.tar.gz" -DataUrl $RightUrl
