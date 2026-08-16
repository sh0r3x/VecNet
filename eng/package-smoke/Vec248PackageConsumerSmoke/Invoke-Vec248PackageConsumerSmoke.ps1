param(
    [Parameter(Mandatory = $true)]
    [string] $PackageSource,

    [Parameter(Mandatory = $true)]
    [string] $ArtifactRoot
)

$ErrorActionPreference = "Stop"

$consumerSource = $PSScriptRoot
$consumerRoot = Join-Path $ArtifactRoot "consumer"
$packagesRoot = Join-Path $ArtifactRoot "packages"
$runRoot = Join-Path $ArtifactRoot "run"
$nugetConfig = Join-Path $ArtifactRoot "NuGet.local.config"

New-Item -ItemType Directory -Force -Path $ArtifactRoot, $consumerRoot, $packagesRoot, $runRoot | Out-Null
Copy-Item -Path (Join-Path $consumerSource "*") -Destination $consumerRoot -Recurse -Force
Get-ChildItem -LiteralPath $consumerRoot -Filter "*.ps1" |
    Remove-Item -Force -ErrorAction SilentlyContinue

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

    if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
        throw "Package file not found: $PackagePath"
    }

    $expanded = Join-Path ([System.IO.Path]::GetTempPath()) ("vecnet-package-smoke-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $expanded | Out-Null
    $zipPath = Join-Path $expanded "package.zip"
    Copy-Item -LiteralPath $PackagePath -Destination $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $expanded -Force

    return $expanded
}

function Get-LocalPackageInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackagePath
    )

    $expanded = Expand-Package -PackagePath $PackagePath

    try {
        $nuspecPath = Get-ChildItem -LiteralPath $expanded -Filter "*.nuspec" | Select-Object -First 1
        if ($null -eq $nuspecPath) {
            throw "No nuspec found in package: $PackagePath"
        }

        [xml] $nuspec = Get-Content -LiteralPath $nuspecPath.FullName -Raw
        $namespace = New-Object System.Xml.XmlNamespaceManager($nuspec.NameTable)
        $namespace.AddNamespace("n", $nuspec.DocumentElement.NamespaceURI)
        $metadata = $nuspec.SelectSingleNode("/n:package/n:metadata", $namespace)
        if ($null -eq $metadata) {
            throw "Nuspec metadata not found in package: $PackagePath"
        }

        return [pscustomobject]@{
            Id = $metadata.SelectSingleNode("n:id", $namespace).InnerText
            Version = $metadata.SelectSingleNode("n:version", $namespace).InnerText
        }
    }
    finally {
        Remove-Item -LiteralPath $expanded -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-RequiredPackageInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackageId
    )

    $packages = @(Get-ChildItem -LiteralPath $PackageSource -Filter "$PackageId.*.nupkg" |
        Where-Object { $_.Name -notlike "*.snupkg" } |
        ForEach-Object { Get-LocalPackageInfo -PackagePath $_.FullName } |
        Where-Object { $_.Id -eq $PackageId })

    if ($packages.Count -ne 1) {
        throw "Expected exactly one $PackageId package in package source '$PackageSource', but found $($packages.Count)."
    }

    return $packages[0]
}

function Set-PackageReferenceVersion {
    param(
        [Parameter(Mandatory = $true)]
        [xml] $Project,

        [Parameter(Mandatory = $true)]
        [string] $PackageId,

        [Parameter(Mandatory = $true)]
        [string] $Version
    )

    $packageReference = @($Project.Project.ItemGroup.PackageReference) |
        Where-Object { $_.Include -eq $PackageId } |
        Select-Object -First 1
    if ($null -eq $packageReference) {
        throw "Consumer project is missing PackageReference to $PackageId."
    }

    $packageReference.Version = $Version
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

function Assert-PackageReferenceRestore {
    param(
        [Parameter(Mandatory = $true)]
        [string] $CorePackageVersion,

        [Parameter(Mandatory = $true)]
        [string] $AdapterPackageVersion
    )

    $projectPath = (Get-ChildItem -LiteralPath $consumerRoot -Filter "*.csproj" | Select-Object -First 1).FullName
    if ([string]::IsNullOrEmpty($projectPath)) {
        throw "Consumer project was not found."
    }

    [xml] $project = Get-Content -LiteralPath $projectPath -Raw
    if ($project.Project.ItemGroup.ProjectReference) {
        throw "Consumer project must not contain ProjectReference items."
    }

    $packageReferences = @($project.Project.ItemGroup.PackageReference)
    $packageIds = $packageReferences | ForEach-Object { $_.Include }
    foreach ($requiredPackageId in @("VecNet", "VecNet.Integration.VectorData")) {
        if ($packageIds -notcontains $requiredPackageId) {
            throw "Consumer project is missing PackageReference to $requiredPackageId."
        }
    }

    $assetsPath = Join-Path $consumerRoot "obj/project.assets.json"
    if (-not (Test-Path -LiteralPath $assetsPath -PathType Leaf)) {
        throw "project.assets.json not found after restore."
    }

    $assets = Get-Content -LiteralPath $assetsPath -Raw | ConvertFrom-Json
    $requiredPackages = @(
        [pscustomobject]@{ Id = "VecNet"; Version = $CorePackageVersion },
        [pscustomobject]@{ Id = "VecNet.Integration.VectorData"; Version = $AdapterPackageVersion }
    )

    foreach ($requiredPackage in $requiredPackages) {
        $library = $assets.libraries.PSObject.Properties |
            Where-Object { $_.Name -eq "$($requiredPackage.Id)/$($requiredPackage.Version)" } |
            Select-Object -First 1
        if ($null -eq $library -or $library.Value.type -ne "package") {
            throw "$($requiredPackage.Id) $($requiredPackage.Version) was not resolved as a NuGet package."
        }
    }

    $projectReferences = @()
    if ($null -ne $assets.project.restore.projectReferences) {
        $projectReferences = @($assets.project.restore.projectReferences.PSObject.Properties)
    }

    if ($projectReferences.Count -ne 0) {
        throw "Consumer restore unexpectedly contains project references."
    }
}

$projectPath = (Get-ChildItem -LiteralPath $consumerRoot -Filter "*.csproj" | Select-Object -First 1).FullName
if ([string]::IsNullOrEmpty($projectPath)) {
    throw "Consumer project was not found."
}

$corePackage = Get-RequiredPackageInfo -PackageId "VecNet"
$adapterPackage = Get-RequiredPackageInfo -PackageId "VecNet.Integration.VectorData"
if ($adapterPackage.Version -ne $corePackage.Version) {
    throw "Consumer smoke expects matching VecNet package versions, but found VecNet $($corePackage.Version) and VecNet.Integration.VectorData $($adapterPackage.Version)."
}

[xml] $project = Get-Content -LiteralPath $projectPath -Raw
Set-PackageReferenceVersion -Project $project -PackageId "VecNet" -Version $corePackage.Version
Set-PackageReferenceVersion -Project $project -PackageId "VecNet.Integration.VectorData" -Version $adapterPackage.Version
$project.Save($projectPath)

Invoke-Checked "dotnet" @("restore", $projectPath, "--configfile", $nugetConfig, "--packages", $packagesRoot, "--verbosity", "quiet") "restore package consumer"
Assert-PackageReferenceRestore -CorePackageVersion $corePackage.Version -AdapterPackageVersion $adapterPackage.Version
Invoke-Checked "dotnet" @("build", $projectPath, "--configuration", "Release", "--no-restore", "--verbosity", "quiet") "build package consumer"
Invoke-Checked "dotnet" @("run", "--project", $projectPath, "--configuration", "Release", "--no-build", "--", $runRoot) "run package consumer"
