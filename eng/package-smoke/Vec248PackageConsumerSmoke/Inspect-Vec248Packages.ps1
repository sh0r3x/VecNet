param(
    [Parameter(Mandatory = $true)]
    [string] $CorePackagePath,

    [Parameter(Mandatory = $true)]
    [string] $AdapterPackagePath
)

$ErrorActionPreference = "Stop"
$expectedVersion = "1.2.0"

function Expand-Package {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackagePath
    )

    if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
        throw "Package file not found: $PackagePath"
    }

    $expanded = Join-Path ([System.IO.Path]::GetTempPath()) ("vec248-package-inspect-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $expanded | Out-Null
    $zipPath = Join-Path $expanded "package.zip"
    Copy-Item -LiteralPath $PackagePath -Destination $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $expanded -Force

    return $expanded
}

function Get-PackageInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackagePath
    )

    $expanded = Expand-Package -PackagePath $PackagePath
    $expandedFullName = (Get-Item -LiteralPath $expanded).FullName.TrimEnd('\', '/')

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

        $files = Get-ChildItem -LiteralPath $expanded -Recurse -File |
            ForEach-Object {
                $_.FullName.Substring($expandedFullName.Length + 1).Replace('\', '/')
            } |
            Where-Object { $_ -ne "package.zip" } |
            Sort-Object

        $dependencies = @()
        foreach ($group in $metadata.SelectNodes("n:dependencies/n:group", $namespace)) {
            foreach ($dependency in $group.SelectNodes("n:dependency", $namespace)) {
                $dependencies += [pscustomobject]@{
                    TargetFramework = $group.targetFramework
                    Id = $dependency.id
                    Version = $dependency.version
                }
            }
        }

        return [pscustomobject]@{
            Expanded = $expanded
            Id = $metadata.SelectSingleNode("n:id", $namespace).InnerText
            Version = $metadata.SelectSingleNode("n:version", $namespace).InnerText
            Description = $metadata.SelectSingleNode("n:description", $namespace).InnerText
            Files = $files
            Dependencies = $dependencies
        }
    }
    catch {
        Remove-Item -LiteralPath $expanded -Recurse -Force -ErrorAction SilentlyContinue
        throw
    }
}

function Assert-NoForbiddenAssets {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackageId,

        [Parameter(Mandatory = $true)]
        [string[]] $Files
    )

    $forbidden = $Files | Where-Object {
        $_.StartsWith("runtimes/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("native/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("build/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("buildTransitive/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("analyzers/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("contentFiles/", [StringComparison]::OrdinalIgnoreCase)
    }

    if ($forbidden) {
        throw "$PackageId has unexpected RID/native/build/analyzer/content assets: $($forbidden -join ', ')"
    }
}

function Assert-RequiredFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackageId,

        [Parameter(Mandatory = $true)]
        [string[]] $Files,

        [Parameter(Mandatory = $true)]
        [string[]] $RequiredFiles
    )

    foreach ($required in $RequiredFiles) {
        if ($Files -notcontains $required) {
            throw "$PackageId is missing required package file: $required"
        }
    }
}

function Test-AcceptedDependencyVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Actual,

        [Parameter(Mandatory = $true)]
        [string] $Expected
    )

    return $Actual -eq $Expected -or $Actual -eq "[$Expected, )" -or $Actual -eq "[$Expected,)"
}

$core = Get-PackageInfo -PackagePath $CorePackagePath
$adapter = Get-PackageInfo -PackagePath $AdapterPackagePath

try {
    if ($core.Id -ne "VecNet") {
        throw "Unexpected core package ID: $($core.Id)"
    }

    if ($core.Version -ne $expectedVersion) {
        throw "Unexpected core package version: $($core.Version)"
    }

    if ($core.Description -notmatch "HNSW approximate search for squared-L2 and cosine distance") {
        throw "Core package description does not mention the admitted HNSW cosine package capability."
    }

    Assert-RequiredFiles -PackageId $core.Id -Files $core.Files -RequiredFiles @(
        "lib/net10.0/VecNet.dll",
        "lib/net10.0/VecNet.xml",
        "README.md",
        "LICENSE"
    )
    Assert-NoForbiddenAssets -PackageId $core.Id -Files $core.Files
    if ($core.Dependencies.Count -ne 0) {
        throw "Core VecNet package should have no dependencies, but found: $($core.Dependencies | ConvertTo-Json -Compress)"
    }

    if ($adapter.Id -ne "VecNet.Integration.VectorData") {
        throw "Unexpected adapter package ID: $($adapter.Id)"
    }

    if ($adapter.Version -ne $expectedVersion) {
        throw "Unexpected adapter package version: $($adapter.Version)"
    }

    if ($adapter.Description -notmatch "exact-flat") {
        throw "Adapter package description must remain exact-flat-only."
    }

    if ($adapter.Description -notmatch "does not provide HNSW VectorData") {
        throw "Adapter package description must not imply HNSW VectorData support."
    }

    Assert-RequiredFiles -PackageId $adapter.Id -Files $adapter.Files -RequiredFiles @(
        "lib/net10.0/VecNet.Integration.VectorData.dll",
        "lib/net10.0/VecNet.Integration.VectorData.xml",
        "README.md",
        "LICENSE"
    )
    Assert-NoForbiddenAssets -PackageId $adapter.Id -Files $adapter.Files

    $adapterDependencies = @($adapter.Dependencies)
    if ($adapterDependencies.Count -ne 2) {
        throw "Adapter package should have exactly two direct dependencies, but found: $($adapterDependencies | ConvertTo-Json -Compress)"
    }

    $vecNetDependency = $adapterDependencies | Where-Object { $_.Id -eq "VecNet" } | Select-Object -First 1
    $vectorDataDependency = $adapterDependencies | Where-Object { $_.Id -eq "Microsoft.Extensions.VectorData.Abstractions" } | Select-Object -First 1
    if ($null -eq $vecNetDependency -or -not (Test-AcceptedDependencyVersion -Actual $vecNetDependency.Version -Expected $expectedVersion)) {
        throw "Adapter package does not depend on accepted VecNet version $expectedVersion."
    }
    if ($null -eq $vectorDataDependency -or -not (Test-AcceptedDependencyVersion -Actual $vectorDataDependency.Version -Expected "10.8.0")) {
        throw "Adapter package does not depend on accepted Microsoft.Extensions.VectorData.Abstractions 10.8.0."
    }

    foreach ($dependency in $adapterDependencies) {
        if ($dependency.Id -notin @("VecNet", "Microsoft.Extensions.VectorData.Abstractions")) {
            throw "Adapter package has unexpected direct dependency: $($dependency.Id)"
        }
    }

    Write-Host "VEC248_CORE_PACKAGE id=$($core.Id) version=$($core.Version)"
    Write-Host "VEC248_CORE_PACKAGE_FILES"
    $core.Files | ForEach-Object { Write-Host "  $_" }
    Write-Host "VEC248_ADAPTER_PACKAGE id=$($adapter.Id) version=$($adapter.Version)"
    Write-Host "VEC248_ADAPTER_PACKAGE_FILES"
    $adapter.Files | ForEach-Object { Write-Host "  $_" }
    Write-Host "VEC248_ADAPTER_DEPENDENCIES"
    $adapterDependencies | ForEach-Object {
        Write-Host "  target=$($_.TargetFramework) id=$($_.Id) version=$($_.Version)"
    }
}
finally {
    Remove-Item -LiteralPath $core.Expanded -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $adapter.Expanded -Recurse -Force -ErrorAction SilentlyContinue
}
