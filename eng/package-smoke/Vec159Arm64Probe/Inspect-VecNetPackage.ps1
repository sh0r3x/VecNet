param(
    [Parameter(Mandatory = $true)]
    [string] $PackagePath
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
    throw "Package file not found: $PackagePath"
}

$expanded = Join-Path ([System.IO.Path]::GetTempPath()) ("vec159-package-inspect-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $expanded | Out-Null

try {
    $zipPath = Join-Path $expanded "package.zip"
    Copy-Item -LiteralPath $PackagePath -Destination $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $expanded -Force

    $nuspecPath = Get-ChildItem -LiteralPath $expanded -Filter "*.nuspec" | Select-Object -First 1
    if ($null -eq $nuspecPath) {
        throw "No nuspec found in package."
    }

    [xml] $nuspec = Get-Content -LiteralPath $nuspecPath.FullName -Raw
    $namespace = New-Object System.Xml.XmlNamespaceManager($nuspec.NameTable)
    $namespace.AddNamespace("n", $nuspec.DocumentElement.NamespaceURI)
    $metadata = $nuspec.SelectSingleNode("/n:package/n:metadata", $namespace)
    if ($null -eq $metadata) {
        throw "Nuspec metadata was not found."
    }

    $id = $metadata.SelectSingleNode("n:id", $namespace).InnerText
    $version = $metadata.SelectSingleNode("n:version", $namespace).InnerText
    $license = $metadata.SelectSingleNode("n:license", $namespace)
    $readme = $metadata.SelectSingleNode("n:readme", $namespace)
    $repository = $metadata.SelectSingleNode("n:repository", $namespace)

    if ($id -ne "VecNet") {
        throw "Unexpected package ID: $id"
    }
    if ($version -ne "1.0.0") {
        throw "Unexpected package version: $version"
    }
    if ($null -eq $license -or $license.type -ne "expression" -or $license.InnerText -ne "MIT") {
        throw "Expected MIT license expression was not found."
    }
    if ($null -eq $readme -or $readme.InnerText -ne "README.md") {
        throw "Expected package README metadata was not found."
    }
    if ($null -eq $repository -or $repository.type -ne "git") {
        throw "Expected git repository metadata was not found."
    }

    $expandedFullName = (Get-Item -LiteralPath $expanded).FullName.TrimEnd('\', '/')
    $files = Get-ChildItem -LiteralPath $expanded -Recurse -File |
        ForEach-Object {
            $_.FullName.Substring($expandedFullName.Length + 1).Replace('\', '/')
        } |
        Where-Object { $_ -ne "package.zip" } |
        Sort-Object

    foreach ($required in @("README.md", "LICENSE", "lib/net10.0/VecNet.dll", "lib/net10.0/VecNet.xml")) {
        if ($files -notcontains $required) {
            throw "Required package file missing: $required"
        }
    }

    $forbidden = $files | Where-Object {
        $_.StartsWith("runtimes/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("native/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("build/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("buildTransitive/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("analyzers/", [StringComparison]::OrdinalIgnoreCase) -or
        $_.StartsWith("contentFiles/", [StringComparison]::OrdinalIgnoreCase)
    }
    if ($forbidden) {
        throw "Unexpected RID/native/build/analyzer/content assets found: $($forbidden -join ', ')"
    }

    $dependencyGroups = $metadata.SelectNodes("n:dependencies/n:group", $namespace)
    foreach ($group in $dependencyGroups) {
        $dependencies = $group.SelectNodes("n:dependency", $namespace)
        if ($dependencies.Count -ne 0) {
            throw "Expected empty dependency groups, but dependencies were found."
        }
    }

    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PackagePath).Hash
    $size = (Get-Item -LiteralPath $PackagePath).Length
    Write-Host "VEC159_PACKAGE id=$id version=$version size=$size sha256=$hash"
    Write-Host "VEC159_PACKAGE_FILES"
    $files | ForEach-Object { Write-Host "  $_" }
}
finally {
    Remove-Item -LiteralPath $expanded -Recurse -Force -ErrorAction SilentlyContinue
}
