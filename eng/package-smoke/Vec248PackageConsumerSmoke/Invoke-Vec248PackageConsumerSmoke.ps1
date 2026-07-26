param(
    [Parameter(Mandatory = $true)]
    [string] $PackageSource,

    [Parameter(Mandatory = $true)]
    [string] $ArtifactRoot
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../../..")
$consumerSource = Join-Path $repoRoot "eng/package-smoke/Vec248PackageConsumerSmoke"
$consumerRoot = Join-Path $ArtifactRoot "consumer"
$packagesRoot = Join-Path $ArtifactRoot "packages-cache"
$runRoot = Join-Path $ArtifactRoot "run"
$nugetConfig = Join-Path $ArtifactRoot "NuGet.local.config"
$expectedVersion = "1.2.0"

New-Item -ItemType Directory -Force -Path $ArtifactRoot, $consumerRoot, $packagesRoot, $runRoot | Out-Null
Copy-Item -Path (Join-Path $consumerSource "*") -Destination $consumerRoot -Recurse -Force
Remove-Item -LiteralPath (Join-Path $consumerRoot "Inspect-Vec248Packages.ps1") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $consumerRoot "Invoke-Vec248PackageConsumerSmoke.ps1") -Force -ErrorAction SilentlyContinue

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

function Assert-PackageReferenceRestore {
    $projectPath = Join-Path $consumerRoot "Vec248PackageConsumer.csproj"
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
    foreach ($requiredPackageId in @("VecNet", "VecNet.Integration.VectorData")) {
        $library = $assets.libraries.PSObject.Properties |
            Where-Object { $_.Name -eq "$requiredPackageId/$expectedVersion" } |
            Select-Object -First 1
        if ($null -eq $library -or $library.Value.type -ne "package") {
            throw "$requiredPackageId $expectedVersion was not resolved as a NuGet package."
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

$projectPath = Join-Path $consumerRoot "Vec248PackageConsumer.csproj"
Invoke-Checked "dotnet" @("restore", $projectPath, "--configfile", $nugetConfig, "--packages", $packagesRoot)
Assert-PackageReferenceRestore
Invoke-Checked "dotnet" @("build", $projectPath, "--configuration", "Release", "--no-restore")
Invoke-Checked "dotnet" @("run", "--project", $projectPath, "--configuration", "Release", "--no-build", "--", $runRoot)
