namespace VecNet.Tests;

public sealed class ExactFlatIndexStorageContractIndependentTests
{
    [Fact]
    public void Save_FileBackedTargetCanLaterBecomeEmptyDirectoryAndSaveSucceeds()
    {
        using TempWorkspace workspace = TempWorkspace.Create();
        string targetPath = Path.Combine(workspace.Path, "index-location");
        File.WriteAllText(targetPath, "caller file");

        var index = CreateIndex();

        Assert.Throws<IOException>(() => index.Save(targetPath));
        Assert.True(File.Exists(targetPath));
        Assert.Equal("caller file", File.ReadAllText(targetPath));

        File.Delete(targetPath);
        Directory.CreateDirectory(targetPath);

        index.Save(targetPath);

        Assert.True(File.Exists(Path.Combine(targetPath, ExactFlatIndexStorage.ManifestFileName)));
        Assert.True(File.Exists(Path.Combine(targetPath, ExactFlatIndexStorage.IdsFileName)));
        Assert.True(File.Exists(Path.Combine(targetPath, ExactFlatIndexStorage.VectorsFileName)));
        Assert.Equal(3, Directory.EnumerateFileSystemEntries(targetPath).Count());
    }

    [Fact]
    public void Save_RepeatedSaveLeavesAllExistingStorageFilesByteIdenticalAndNoTemps()
    {
        using TempWorkspace workspace = TempWorkspace.Create();
        string targetPath = Path.Combine(workspace.Path, "saved-index");
        var index = CreateIndex();
        index.Save(targetPath);

        byte[] manifestBefore = File.ReadAllBytes(Path.Combine(targetPath, ExactFlatIndexStorage.ManifestFileName));
        byte[] idsBefore = File.ReadAllBytes(Path.Combine(targetPath, ExactFlatIndexStorage.IdsFileName));
        byte[] vectorsBefore = File.ReadAllBytes(Path.Combine(targetPath, ExactFlatIndexStorage.VectorsFileName));

        Assert.Throws<IOException>(() => index.Save(targetPath));

        Assert.Equal(manifestBefore, File.ReadAllBytes(Path.Combine(targetPath, ExactFlatIndexStorage.ManifestFileName)));
        Assert.Equal(idsBefore, File.ReadAllBytes(Path.Combine(targetPath, ExactFlatIndexStorage.IdsFileName)));
        Assert.Equal(vectorsBefore, File.ReadAllBytes(Path.Combine(targetPath, ExactFlatIndexStorage.VectorsFileName)));
        Assert.DoesNotContain(Directory.EnumerateFileSystemEntries(targetPath), static path => Path.GetFileName(path).Contains(".tmp-", StringComparison.Ordinal));
    }

    [Fact]
    public void Save_FailedInternalSaveForCreatedDirectoryRemovesTargetButPreservesParentContents()
    {
        using TempWorkspace workspace = TempWorkspace.Create();
        string markerPath = Path.Combine(workspace.Path, "parent-marker.txt");
        File.WriteAllText(markerPath, "keep");
        string targetPath = Path.Combine(workspace.Path, "new-index");

        Assert.Throws<InvalidOperationException>(() =>
            ExactFlatIndexStorage.Save(
                targetPath,
                dimension: 2,
                VectorMetric.SquaredEuclidean,
                [42UL],
                []));

        Assert.False(Directory.Exists(targetPath));
        Assert.True(File.Exists(markerPath));
        Assert.Equal("keep", File.ReadAllText(markerPath));
        Assert.Equal(["parent-marker.txt"], Directory.EnumerateFileSystemEntries(workspace.Path).Select(Path.GetFileName).Order());
    }

    [Fact]
    public void Save_FailedInternalSaveInExistingEmptyDirectoryIsRepeatableAndLeavesDirectoryReusable()
    {
        using TempWorkspace workspace = TempWorkspace.Create();
        string targetPath = Path.Combine(workspace.Path, "existing-empty");
        Directory.CreateDirectory(targetPath);

        for (int attempt = 0; attempt < 2; attempt++)
        {
            Assert.Throws<InvalidOperationException>(() =>
                ExactFlatIndexStorage.Save(
                    targetPath,
                    dimension: 2,
                    VectorMetric.SquaredEuclidean,
                    [42UL],
                    []));

            Assert.True(Directory.Exists(targetPath));
            Assert.Empty(Directory.EnumerateFileSystemEntries(targetPath));
        }

        CreateIndex().Save(targetPath);

        Assert.Equal(
            [
                ExactFlatIndexStorage.IdsFileName,
                ExactFlatIndexStorage.ManifestFileName,
                ExactFlatIndexStorage.VectorsFileName
            ],
            Directory.EnumerateFileSystemEntries(targetPath).Select(Path.GetFileName).Order());
    }

    [Fact]
    public void OpenReadOnly_InvalidStorageLocationsAreClassifiedWithoutMutation()
    {
        using TempWorkspace workspace = TempWorkspace.Create();

        string fileBackedPath = Path.Combine(workspace.Path, "not-a-directory");
        File.WriteAllText(fileBackedPath, "file-backed target");
        IOException fileException = Assert.Throws<IOException>(() => ExactFlatIndex.OpenReadOnly(fileBackedPath));
        Assert.Contains("file", fileException.Message, StringComparison.OrdinalIgnoreCase);
        Assert.True(File.Exists(fileBackedPath));
        Assert.Equal("file-backed target", File.ReadAllText(fileBackedPath));

        string missingPath = Path.Combine(workspace.Path, "missing-index");
        Assert.Throws<DirectoryNotFoundException>(() => ExactFlatIndex.OpenReadOnly(missingPath));
        Assert.False(Directory.Exists(missingPath));

        string emptyDirectoryPath = Path.Combine(workspace.Path, "empty-index");
        Directory.CreateDirectory(emptyDirectoryPath);
        Assert.Throws<FileNotFoundException>(() => ExactFlatIndex.OpenReadOnly(emptyDirectoryPath));
        Assert.Empty(Directory.EnumerateFileSystemEntries(emptyDirectoryPath));
    }

    private static ExactFlatIndex CreateIndex()
    {
        var index = new ExactFlatIndex(2, VectorMetric.SquaredEuclidean);
        index.Add(20, [1f, 0.25f]);
        index.Add(10, [-1f, 0.5f]);
        return index;
    }

    private sealed class TempWorkspace : IDisposable
    {
        private TempWorkspace(string path) => Path = path;

        public string Path { get; }

        public static TempWorkspace Create()
        {
            string path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "VecNet.Tests",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return new TempWorkspace(path);
        }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }
}
