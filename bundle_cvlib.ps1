param(
    [string]$Config = "Release",
    [string]$CvlibSourceDir = "",
    [string]$Generator = "",
    [string]$Platform = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($CvlibSourceDir -eq "") {
    $WorkspaceDir = Split-Path -Parent $ScriptDir
    $CvlibSourceDir = Join-Path $WorkspaceDir "cvlib\cpp"
}

if (!(Test-Path (Join-Path $CvlibSourceDir "CMakeLists.txt"))) {
    Write-Error "cvlib source not found: $CvlibSourceDir"
    exit 1
}

if ($Platform -eq "") {
    $Platform = "linux"
    if ($env:OS -eq "Windows_NT") {
        $Platform = "msvc"
    } elseif ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
            [System.Runtime.InteropServices.OSPlatform]::OSX)) {
        $Platform = "macos"
    }
}

$ConfigLower = $Config.ToLowerInvariant()
$BuildDir = Join-Path $ScriptDir "build\cvlib_package\$Platform\$ConfigLower"
$ConfigureArgs = @(
    "-S", $CvlibSourceDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCVLIB_BUILD_TESTS=OFF",
    "-DCVLIB_BUILD_PYTHON=OFF"
)
if ($Platform -match "^(linux|unix)$") {
    $LinuxCxxFlags = $env:CXXFLAGS
    if ($null -eq $LinuxCxxFlags) {
        $LinuxCxxFlags = ""
    }
    $LinuxCxxFlags = "$LinuxCxxFlags -Wno-unused-function".Trim()
    $ConfigureArgs += "-DCMAKE_CXX_FLAGS=$LinuxCxxFlags"
}
if ($Generator -ne "") {
    $ConfigureArgs = @("-G", $Generator) + $ConfigureArgs
}

& cmake --log-level=WARNING @ConfigureArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$BundleRoot = Join-Path $ScriptDir "thirdparty\cvlib"
$IncludeDest = Join-Path $BundleRoot "include"
if (Test-Path $IncludeDest) {
    Remove-Item -LiteralPath $IncludeDest -Recurse -Force
}
New-Item -ItemType Directory -Force $IncludeDest | Out-Null
$CvlibIncludeSource = Join-Path $CvlibSourceDir "include\cvlib"
if (!(Test-Path $CvlibIncludeSource)) {
    $CvlibIncludeSource = Join-Path $CvlibSourceDir "include"
}
Copy-Item -Path (Join-Path $CvlibIncludeSource "*") `
    -Destination $IncludeDest -Recurse -Force
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
# Rewrite "cvlib/<path>" includes to includer-relative paths ("../<path>"
# from one level down) so quoted-include resolution stays inside the
# bundle even when a consumer include directory shadows names such as
# types.h earlier in the include search order.
$IncludeDestFull = (Get-Item $IncludeDest).FullName
Get-ChildItem -Path $IncludeDest -Recurse -Filter "*.h" | ForEach-Object {
    $RelativeDir = $_.DirectoryName.Substring($IncludeDestFull.Length).Trim('\', '/')
    $Prefix = ""
    if ($RelativeDir -ne "") {
        $Depth = ($RelativeDir -split '[\\/]').Count
        $Prefix = "../" * $Depth
    }
    $HeaderText = [System.IO.File]::ReadAllText($_.FullName)
    $HeaderText = $HeaderText.Replace('#include "cvlib/', "#include `"$Prefix")
    [System.IO.File]::WriteAllText($_.FullName, $HeaderText, $Utf8NoBom)
}

$LibraryNames = @()
if ($Platform -match "^(msvc|windows)$") {
    $LibraryNames = @("cvlib_core.lib")
} elseif ($Platform -match "^(linux|unix)$") {
    $LibraryNames = @("libcvlib_core.a", "libcvlib_core.so")
} elseif ($Platform -match "^(macos|darwin)$") {
    $LibraryNames = @("libcvlib_core.a", "libcvlib_core.dylib")
} else {
    $LibraryNames = @(
        "cvlib_core.lib",
        "libcvlib_core.a",
        "libcvlib_core.so",
        "libcvlib_core.dylib"
    )
}

$Libraries = @(Get-ChildItem -Recurse -File $BuildDir |
    Where-Object { $LibraryNames -contains $_.Name } |
    Sort-Object FullName |
    Select-Object)
if ($Libraries.Count -eq 0) {
    Write-Error "cvlib library for platform '$Platform' was not produced under $BuildDir"
    exit 1
}

$LibDest = Join-Path $BundleRoot "lib\$Platform\$ConfigLower"
New-Item -ItemType Directory -Force $LibDest | Out-Null
foreach ($LibraryName in $LibraryNames) {
    $ExistingLibrary = Join-Path $LibDest $LibraryName
    if (Test-Path $ExistingLibrary) {
        Remove-Item -LiteralPath $ExistingLibrary -Force
    }
}
foreach ($Library in $Libraries) {
    Copy-Item -LiteralPath $Library.FullName -Destination $LibDest -Force
}

Write-Host "cvlib bundled:"
Write-Host "  headers: $IncludeDest"
foreach ($Library in $Libraries) {
    Write-Host "  library: $(Join-Path $LibDest $Library.Name)"
}
