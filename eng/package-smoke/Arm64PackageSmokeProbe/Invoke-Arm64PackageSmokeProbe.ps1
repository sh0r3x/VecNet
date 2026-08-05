param(
    [Parameter(Mandatory = $true)]
    [string] $PackagePath,

    [Parameter(Mandatory = $true)]
    [string] $Rid,

    [Parameter(Mandatory = $true)]
    [string] $PlatformName,

    [Parameter(Mandatory = $true)]
    [string] $RunnerLabel,

    [Parameter(Mandatory = $true)]
    [string] $PackageVersion,

    [switch] $RequireArm64
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../../..")
$consumerSource = Join-Path $repoRoot "eng/package-smoke/Arm64PackageSmokeConsumer"
$probeRoot = Join-Path $env:RUNNER_TEMP "arm64-package-smoke"
$consumerRoot = Join-Path $probeRoot "consumer"
$packageSource = Split-Path -Parent (Resolve-Path -LiteralPath $PackagePath)
$packagesRoot = Join-Path $probeRoot "packages"
$resultsRoot = Join-Path $probeRoot ("results-" + [Guid]::NewGuid().ToString("N"))
$localConfig = Join-Path $probeRoot "NuGet.local.config"
$runtimeConfig = Join-Path $probeRoot "NuGet.runtime.config"
$summaryPath = if ($env:GITHUB_STEP_SUMMARY) { $env:GITHUB_STEP_SUMMARY } else { Join-Path $resultsRoot "summary.md" }
$statuses = [ordered]@{}

New-Item -ItemType Directory -Force -Path $probeRoot, $consumerRoot, $packagesRoot, $resultsRoot | Out-Null
Copy-Item -Path (Join-Path $consumerSource "*") -Destination $consumerRoot -Recurse -Force
$consumerProjectPath = Join-Path $consumerRoot "Arm64PackageSmokeConsumer.csproj"
[xml] $consumerProject = Get-Content -LiteralPath $consumerProjectPath -Raw
$vecNetReference = $consumerProject.Project.ItemGroup.PackageReference |
    Where-Object { $_.Include -eq "VecNet" } |
    Select-Object -First 1
if ($null -eq $vecNetReference) {
    throw "Arm64 package-smoke consumer project does not contain a VecNet PackageReference."
}
$vecNetReference.Version = $PackageVersion
$consumerProject.Save($consumerProjectPath)

@"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="vecnet-local" value="$packageSource" />
  </packageSources>
</configuration>
"@ | Set-Content -LiteralPath $localConfig -Encoding UTF8

@"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="vecnet-local" value="$packageSource" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
"@ | Set-Content -LiteralPath $runtimeConfig -Encoding UTF8

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    Write-Host ">> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Invoke-ProbeStep {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,

        [Parameter(Mandatory = $true)]
        [scriptblock] $Body
    )

    Write-Host "::group::$Name"
    try {
        & $Body
        $statuses[$Name] = "passed"
        Write-Host "ARM64_PACKAGE_SMOKE_DISPOSITION platform=$PlatformName row=`"$Name`" status=passed"
    }
    catch {
        $message = $_.Exception.Message
        $statuses[$Name] = "failed: $message"
        Write-Host "ARM64_PACKAGE_SMOKE_DISPOSITION platform=$PlatformName row=`"$Name`" status=failed reason=`"$message`""
        Write-Warning $message
    }
    finally {
        Write-Host "::endgroup::"
    }
}

function Invoke-Consumer {
    param(
        [Parameter(Mandatory = $true)]
        [string] $ExecutablePath,

        [Parameter(Mandatory = $true)]
        [string] $Mode,

        [Parameter(Mandatory = $true)]
        [string] $ArtifactRoot
    )

    $args = @("--mode", $Mode, "--artifact-root", $ArtifactRoot, "--runner-label", $RunnerLabel)
    if ($RequireArm64) {
        $args += "--require-arm64"
    }

    Invoke-Checked -FilePath $ExecutablePath -Arguments $args
}

function Get-AppHostPath {
    param([Parameter(Mandatory = $true)][string] $Directory)

    $windowsPath = Join-Path $Directory "Arm64PackageSmokeConsumer.exe"
    $unixPath = Join-Path $Directory "Arm64PackageSmokeConsumer"
    if (Test-Path -LiteralPath $windowsPath -PathType Leaf) {
        return $windowsPath
    }
    if (Test-Path -LiteralPath $unixPath -PathType Leaf) {
        return $unixPath
    }

    return Join-Path $Directory "Arm64PackageSmokeConsumer.dll"
}

function Assert-PackageReferenceRestore {
    $assetsPath = Join-Path $consumerRoot "obj/project.assets.json"
    if (-not (Test-Path -LiteralPath $assetsPath -PathType Leaf)) {
        throw "project.assets.json not found after restore."
    }

    $assets = Get-Content -LiteralPath $assetsPath -Raw | ConvertFrom-Json
    $vecnetLibrary = $assets.libraries.PSObject.Properties | Where-Object { $_.Name -like "VecNet/*" } | Select-Object -First 1
    if ($null -eq $vecnetLibrary -or $vecnetLibrary.Value.type -ne "package") {
        throw "VecNet was not resolved as a NuGet package."
    }

    if ($assets.project.restore.projectReferences.PSObject.Properties.Count -ne 0) {
        throw "Consumer restore unexpectedly contains project references."
    }
}

$projectPath = $consumerProjectPath
$arm64Flag = if ($RequireArm64) { @("--require-arm64") } else { @() }

Invoke-ProbeStep -Name "$PlatformName JIT package smoke" -Body {
    Invoke-Checked "dotnet" @("restore", $projectPath, "--configfile", $localConfig, "--packages", $packagesRoot)
    Assert-PackageReferenceRestore
    Invoke-Checked "dotnet" @("build", $projectPath, "--configuration", "Release", "--no-restore")
    $runArgs = @(
        "run", "--project", $projectPath,
        "--configuration", "Release",
        "--no-build",
        "--",
        "--mode", "jit",
        "--artifact-root", (Join-Path $resultsRoot "jit durable path with spaces"),
        "--runner-label", $RunnerLabel
    )
    $runArgs += $arm64Flag
    Invoke-Checked "dotnet" $runArgs
}

Invoke-ProbeStep -Name "$PlatformName copied deployment" -Body {
    $publishDir = Join-Path $probeRoot "framework-dependent-publish"
    $copiedDir = Join-Path $probeRoot "copied publish with spaces"
    Invoke-Checked "dotnet" @("publish", $projectPath, "--configuration", "Release", "--no-restore", "--output", $publishDir)
    if (Test-Path -LiteralPath $copiedDir) {
        Remove-Item -LiteralPath $copiedDir -Recurse -Force
    }
    Copy-Item -LiteralPath $publishDir -Destination $copiedDir -Recurse -Force
    $appHost = Get-AppHostPath -Directory $copiedDir
    Push-Location $copiedDir
    try {
        if ($appHost.EndsWith(".dll", [StringComparison]::OrdinalIgnoreCase)) {
            $copiedArgs = @($appHost, "--mode", "copied", "--artifact-root", (Join-Path $resultsRoot "copied durable path with spaces"), "--runner-label", $RunnerLabel)
            $copiedArgs += $arm64Flag
            Invoke-Checked "dotnet" $copiedArgs
        }
        else {
            Invoke-Consumer -ExecutablePath $appHost -Mode "copied" -ArtifactRoot (Join-Path $resultsRoot "copied durable path with spaces")
        }
    }
    finally {
        Pop-Location
    }
}

Invoke-ProbeStep -Name "$PlatformName trimming" -Body {
    $trimmedDir = Join-Path $probeRoot "trimmed-publish"
    Invoke-Checked "dotnet" @(
        "publish", $projectPath,
        "--configuration", "Release",
        "--runtime", $Rid,
        "--self-contained", "true",
        "--packages", $packagesRoot,
        "--configfile", $runtimeConfig,
        "--output", $trimmedDir,
        "-p:PublishTrimmed=true",
        "-p:TreatWarningsAsErrors=true"
    )
    $appHost = Get-AppHostPath -Directory $trimmedDir
    Invoke-Consumer -ExecutablePath $appHost -Mode "trimmed" -ArtifactRoot (Join-Path $resultsRoot "trimmed durable path with spaces")
}

Invoke-ProbeStep -Name "$PlatformName NativeAOT" -Body {
    $aotDir = Join-Path $probeRoot "nativeaot-publish"
    Invoke-Checked "dotnet" @(
        "publish", $projectPath,
        "--configuration", "Release",
        "--runtime", $Rid,
        "--self-contained", "true",
        "--packages", $packagesRoot,
        "--configfile", $runtimeConfig,
        "--output", $aotDir,
        "-p:PublishAot=true",
        "-p:TreatWarningsAsErrors=true"
    )
    $appHost = Get-AppHostPath -Directory $aotDir
    Invoke-Consumer -ExecutablePath $appHost -Mode "nativeaot" -ArtifactRoot (Join-Path $resultsRoot "nativeaot durable path with spaces")
}

Add-Content -LiteralPath $summaryPath -Value "## Arm64 package smoke - $PlatformName"
Add-Content -LiteralPath $summaryPath -Value ""
Add-Content -LiteralPath $summaryPath -Value "| Row | Disposition |"
Add-Content -LiteralPath $summaryPath -Value "| --- | --- |"
foreach ($entry in $statuses.GetEnumerator()) {
    Add-Content -LiteralPath $summaryPath -Value "| $($entry.Key) | $($entry.Value) |"
}

$failed = $statuses.GetEnumerator() | Where-Object { $_.Value -ne "passed" }
if ($failed) {
    throw "One or more Arm64 package-smoke $PlatformName rows failed or were blocked. See dispositions above."
}
