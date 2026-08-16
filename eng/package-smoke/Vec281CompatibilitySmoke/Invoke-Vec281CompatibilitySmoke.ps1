param(
    [Parameter(Mandatory = $true)]
    [string] $PackageSource,

    [Parameter(Mandatory = $true)]
    [string] $ArtifactRoot
)

$ErrorActionPreference = "Stop"

$baselineProject = (Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "BaselineWriter") -Filter "*.csproj" | Select-Object -First 1).FullName
$currentProject = (Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "CurrentReader") -Filter "*.csproj" | Select-Object -First 1).FullName
if ([string]::IsNullOrEmpty($baselineProject) -or [string]::IsNullOrEmpty($currentProject)) {
    throw "Compatibility smoke projects were not found."
}

$packagesRoot = Join-Path $ArtifactRoot "packages"
$nugetConfig = Join-Path $ArtifactRoot "NuGet.local.config"

New-Item -ItemType Directory -Force -Path $ArtifactRoot, $packagesRoot | Out-Null

@"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="vecnet-local" value="$(Resolve-Path -LiteralPath $PackageSource)" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
"@ | Set-Content -LiteralPath $nugetConfig -Encoding UTF8

function Expand-Package {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackagePath
    )

    $expanded = Join-Path ([System.IO.Path]::GetTempPath()) ("vecnet-compatibility-smoke-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $expanded | Out-Null
    $zipPath = Join-Path $expanded "package.zip"
    Copy-Item -LiteralPath $PackagePath -Destination $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $expanded -Force
    return $expanded
}

function Get-LocalPackageVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackageId
    )

    $packages = @(Get-ChildItem -LiteralPath $PackageSource -Filter "$PackageId.*.nupkg" |
        Where-Object { $_.Name -notlike "*.snupkg" } |
        ForEach-Object {
            $expanded = Expand-Package -PackagePath $_.FullName
            try {
                $nuspecPath = Get-ChildItem -LiteralPath $expanded -Filter "*.nuspec" | Select-Object -First 1
                if ($null -eq $nuspecPath) {
                    throw "No nuspec found in package: $($_.FullName)"
                }

                [xml] $nuspec = Get-Content -LiteralPath $nuspecPath.FullName -Raw
                $namespace = New-Object System.Xml.XmlNamespaceManager($nuspec.NameTable)
                $namespace.AddNamespace("n", $nuspec.DocumentElement.NamespaceURI)
                $metadata = $nuspec.SelectSingleNode("/n:package/n:metadata", $namespace)
                [pscustomobject]@{
                    Id = $metadata.SelectSingleNode("n:id", $namespace).InnerText
                    Version = $metadata.SelectSingleNode("n:version", $namespace).InnerText
                }
            }
            finally {
                Remove-Item -LiteralPath $expanded -Recurse -Force -ErrorAction SilentlyContinue
            }
        } |
        Where-Object { $_.Id -eq $PackageId })

    if ($packages.Count -ne 1) {
        throw "Expected exactly one $PackageId package in package source '$PackageSource', but found $($packages.Count)."
    }

    return $packages[0].Version
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments,

        [Parameter(Mandatory = $true)]
        [string] $DisplayName
    )

    Write-Host ">> $DisplayName"
    $output = & $FilePath @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        $output | ForEach-Object { Write-Host $_ }
        throw "Command failed with exit code ${LASTEXITCODE}: $DisplayName"
    }

    $output |
        Where-Object { $_ -match '^(PACKAGE_|COMPATIBILITY_)' } |
        ForEach-Object { Write-Host $_ }
}

$currentVersion = Get-LocalPackageVersion -PackageId "VecNet"
foreach ($baselineVersion in @("1.0.0", "1.3.1")) {
    Invoke-Checked "dotnet" @(
        "restore",
        $baselineProject,
        "--configfile",
        $nugetConfig,
        "--packages",
        $packagesRoot,
        "-p:VecNetPackageVersion=$baselineVersion",
        "--verbosity",
        "quiet") "restore baseline package $baselineVersion"
    Invoke-Checked "dotnet" @(
        "run",
        "--project",
        $baselineProject,
        "--configuration",
        "Release",
        "--no-restore",
        "-p:VecNetPackageVersion=$baselineVersion",
        "--",
        $ArtifactRoot,
        $baselineVersion) "write baseline package $baselineVersion"
}

Invoke-Checked "dotnet" @(
    "restore",
    $currentProject,
    "--configfile",
    $nugetConfig,
    "--packages",
    $packagesRoot,
    "-p:VecNetPackageVersion=$currentVersion",
    "--verbosity",
    "quiet") "restore current package"
Invoke-Checked "dotnet" @(
    "run",
    "--project",
    $currentProject,
    "--configuration",
    "Release",
    "--no-restore",
    "-p:VecNetPackageVersion=$currentVersion",
    "--",
    $ArtifactRoot) "read compatibility baselines"

$exclusions = @(Get-ChildItem -LiteralPath (Join-Path $ArtifactRoot "baselines") -Filter "excluded-scenarios.txt" -Recurse |
    ForEach-Object { Get-Content -LiteralPath $_.FullName } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
foreach ($exclusion in $exclusions) {
    Write-Host "COMPATIBILITY_EXCLUSION $exclusion"
}

Write-Host "COMPATIBILITY_SMOKE_PASSED current=$currentVersion"
