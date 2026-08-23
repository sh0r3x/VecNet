param(
    [Parameter(Mandatory = $true)]
    [string] $CorePackagePath,

    [Parameter(Mandatory = $true)]
    [string] $AdapterPackagePath,

    [string] $ExpectedPackageVersion,

    [string] $CoreSymbolPackagePath,

    [string] $AdapterSymbolPackagePath,

    [string[]] $ForbiddenPayloadText
)

$ErrorActionPreference = "Stop"

function Expand-Package {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackagePath
    )

    if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
        throw "Package file not found: $PackagePath"
    }

    $expanded = Join-Path ([System.IO.Path]::GetTempPath()) ("vecnet-package-inspect-" + [Guid]::NewGuid().ToString("N"))
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

        $projectUrl = $metadata.SelectSingleNode("n:projectUrl", $namespace)
        $license = $metadata.SelectSingleNode("n:license", $namespace)
        $repository = $metadata.SelectSingleNode("n:repository", $namespace)
        $projectUrlValue = if ($null -eq $projectUrl) { $null } else { $projectUrl.InnerText }
        $licenseTypeValue = if ($null -eq $license) { $null } else { $license.GetAttribute("type") }
        $licenseValue = if ($null -eq $license) { $null } else { $license.InnerText }
        $repositoryTypeValue = if ($null -eq $repository) { $null } else { $repository.GetAttribute("type") }
        $repositoryUrlValue = if ($null -eq $repository) { $null } else { $repository.GetAttribute("url") }
        $repositoryCommitValue = if ($null -eq $repository) { $null } else { $repository.GetAttribute("commit") }

        return [pscustomobject]@{
            Expanded = $expanded
            Id = $metadata.SelectSingleNode("n:id", $namespace).InnerText
            Version = $metadata.SelectSingleNode("n:version", $namespace).InnerText
            Description = $metadata.SelectSingleNode("n:description", $namespace).InnerText
            ProjectUrl = $projectUrlValue
            LicenseType = $licenseTypeValue
            License = $licenseValue
            RepositoryType = $repositoryTypeValue
            RepositoryUrl = $repositoryUrlValue
            RepositoryCommit = $repositoryCommitValue
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

function Assert-PackageMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [object] $Package,

        [bool] $RequireLicense = $true
    )

    if ($Package.ProjectUrl -ne "https://github.com/sh0r3x/VecNet") {
        throw "$($Package.Id) has unexpected project URL: $($Package.ProjectUrl)"
    }

    if ($Package.RepositoryType -ne "git") {
        throw "$($Package.Id) has unexpected repository type: $($Package.RepositoryType)"
    }

    if ($Package.RepositoryUrl -ne "https://github.com/sh0r3x/VecNet.git") {
        throw "$($Package.Id) has unexpected repository URL: $($Package.RepositoryUrl)"
    }

    if ([string]::IsNullOrWhiteSpace($Package.RepositoryCommit)) {
        throw "$($Package.Id) package did not emit repository commit metadata."
    }

    if ($RequireLicense -and ($Package.LicenseType -ne "expression" -or $Package.License -ne "MIT")) {
        throw "$($Package.Id) has unexpected license metadata: type=$($Package.LicenseType) value=$($Package.License)"
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

function Get-DefaultSymbolPackagePath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackagePath
    )

    return [System.IO.Path]::ChangeExtension($PackagePath, ".snupkg")
}

function Assert-SymbolPackage {
    param(
        [Parameter(Mandatory = $true)]
        [string] $PackageId,

        [Parameter(Mandatory = $true)]
        [string] $PackagePath,

        [Parameter(Mandatory = $true)]
        [string[]] $RequiredFiles
    )

    if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
        throw "$PackageId symbol package was not found: $PackagePath"
    }

    $symbol = Get-PackageInfo -PackagePath $PackagePath
    try {
        if ($symbol.Id -ne $PackageId) {
            throw "Unexpected symbol package ID for ${PackageId}: $($symbol.Id)"
        }

        Assert-PackageMetadata -Package $symbol -RequireLicense $false
        Assert-RequiredFiles -PackageId $symbol.Id -Files $symbol.Files -RequiredFiles $RequiredFiles
        Assert-NoForbiddenAssets -PackageId $symbol.Id -Files $symbol.Files

        return $symbol
    }
    catch {
        Remove-Item -LiteralPath $symbol.Expanded -Recurse -Force -ErrorAction SilentlyContinue
        throw
    }
}

function Assert-NoForbiddenPayloadText {
    param(
        [Parameter(Mandatory = $true)]
        [string] $ExpandedRoot,

        [Parameter(Mandatory = $true)]
        [string] $PackageId,

        [string[]] $Patterns
    )

    if ($null -eq $Patterns -or $Patterns.Count -eq 0) {
        return
    }

    foreach ($file in Get-ChildItem -LiteralPath $ExpandedRoot -Recurse -File) {
        if ($file.Name -eq "package.zip") {
            continue
        }

        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
        $utf8 = [System.Text.Encoding]::UTF8.GetString($bytes)
        $unicode = [System.Text.Encoding]::Unicode.GetString($bytes)
        foreach ($pattern in $Patterns) {
            if ([string]::IsNullOrWhiteSpace($pattern)) {
                continue
            }

            if ($utf8.IndexOf($pattern, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
                $unicode.IndexOf($pattern, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                throw "$PackageId payload contains forbidden text '$pattern' in $($file.Name)."
            }
        }
    }
}

$core = Get-PackageInfo -PackagePath $CorePackagePath
$adapter = Get-PackageInfo -PackagePath $AdapterPackagePath
$coreSymbolPackage = $null
$adapterSymbolPackage = $null

try {
    $expectedVersion = if ([string]::IsNullOrWhiteSpace($ExpectedPackageVersion)) {
        $core.Version
    }
    else {
        $ExpectedPackageVersion
    }

    if ($core.Id -ne "VecNet") {
        throw "Unexpected core package ID: $($core.Id)"
    }

    if ($core.Version -ne $expectedVersion) {
        throw "Unexpected core package version: $($core.Version)"
    }

    Assert-PackageMetadata -Package $core

    if ($core.Description -notmatch "HNSW approximate search for squared-L2, inner-product, and cosine workloads") {
        throw "Core package description does not mention the admitted HNSW metric package capabilities."
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

    Assert-PackageMetadata -Package $adapter

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

    $coreSymbolPath = if ([string]::IsNullOrWhiteSpace($CoreSymbolPackagePath)) {
        Get-DefaultSymbolPackagePath -PackagePath $CorePackagePath
    }
    else {
        $CoreSymbolPackagePath
    }
    $adapterSymbolPath = if ([string]::IsNullOrWhiteSpace($AdapterSymbolPackagePath)) {
        Get-DefaultSymbolPackagePath -PackagePath $AdapterPackagePath
    }
    else {
        $AdapterSymbolPackagePath
    }

    $coreSymbolPackage = Assert-SymbolPackage -PackageId "VecNet" -PackagePath $coreSymbolPath -RequiredFiles @(
        "lib/net10.0/VecNet.pdb"
    )
    $adapterSymbolPackage = Assert-SymbolPackage -PackageId "VecNet.Integration.VectorData" -PackagePath $adapterSymbolPath -RequiredFiles @(
        "lib/net10.0/VecNet.Integration.VectorData.pdb"
    )

    foreach ($payload in @($core, $adapter, $coreSymbolPackage, $adapterSymbolPackage)) {
        Assert-NoForbiddenPayloadText -ExpandedRoot $payload.Expanded -PackageId $payload.Id -Patterns $ForbiddenPayloadText
    }

    Write-Host "CORE_PACKAGE id=$($core.Id) version=$($core.Version)"
    Write-Host "CORE_PACKAGE_METADATA projectUrl=$($core.ProjectUrl) repositoryType=$($core.RepositoryType) repositoryUrl=$($core.RepositoryUrl) repositoryCommit=$($core.RepositoryCommit) license=$($core.LicenseType):$($core.License)"
    Write-Host "CORE_PACKAGE_FILES"
    $core.Files | ForEach-Object { Write-Host "  $_" }
    Write-Host "CORE_SYMBOL_PACKAGE id=$($coreSymbolPackage.Id) version=$($coreSymbolPackage.Version)"
    Write-Host "CORE_SYMBOL_PACKAGE_FILES"
    $coreSymbolPackage.Files | ForEach-Object { Write-Host "  $_" }
    Write-Host "ADAPTER_PACKAGE id=$($adapter.Id) version=$($adapter.Version)"
    Write-Host "ADAPTER_PACKAGE_METADATA projectUrl=$($adapter.ProjectUrl) repositoryType=$($adapter.RepositoryType) repositoryUrl=$($adapter.RepositoryUrl) repositoryCommit=$($adapter.RepositoryCommit) license=$($adapter.LicenseType):$($adapter.License)"
    Write-Host "ADAPTER_PACKAGE_FILES"
    $adapter.Files | ForEach-Object { Write-Host "  $_" }
    Write-Host "ADAPTER_SYMBOL_PACKAGE id=$($adapterSymbolPackage.Id) version=$($adapterSymbolPackage.Version)"
    Write-Host "ADAPTER_SYMBOL_PACKAGE_FILES"
    $adapterSymbolPackage.Files | ForEach-Object { Write-Host "  $_" }
    Write-Host "ADAPTER_DEPENDENCIES"
    $adapterDependencies | ForEach-Object {
        Write-Host "  target=$($_.TargetFramework) id=$($_.Id) version=$($_.Version)"
    }
}
finally {
    Remove-Item -LiteralPath $core.Expanded -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $adapter.Expanded -Recurse -Force -ErrorAction SilentlyContinue
    if ($null -ne $coreSymbolPackage) {
        Remove-Item -LiteralPath $coreSymbolPackage.Expanded -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $adapterSymbolPackage) {
        Remove-Item -LiteralPath $adapterSymbolPackage.Expanded -Recurse -Force -ErrorAction SilentlyContinue
    }
}
