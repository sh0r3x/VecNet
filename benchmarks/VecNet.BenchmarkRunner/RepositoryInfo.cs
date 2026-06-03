using System.Diagnostics;

namespace VecNet.BenchmarkRunner;

public sealed record RepositoryInfo(string? Commit, string? Branch, bool Dirty)
{
    public static RepositoryInfo Create()
    {
        string? commit = RunGit("rev-parse", "HEAD");
        string? branch = RunGit("branch", "--show-current");
        string? status = RunGit("status", "--porcelain=v1", "--untracked-files=all");
        return new RepositoryInfo(
            string.IsNullOrWhiteSpace(commit) ? null : commit,
            string.IsNullOrWhiteSpace(branch) ? null : branch,
            !string.IsNullOrWhiteSpace(status));
    }

    private static string? RunGit(params string[] arguments)
    {
        try
        {
            var startInfo = new ProcessStartInfo("git")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };

            foreach (string argument in arguments)
            {
                startInfo.ArgumentList.Add(argument);
            }

            using Process process = Process.Start(startInfo)!;
            string output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit(2000);

            return process.ExitCode == 0 ? output : null;
        }
        catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }
}
