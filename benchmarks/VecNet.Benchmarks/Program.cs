using BenchmarkDotNet.Running;

namespace VecNet.Benchmarks;

internal static class Program
{
    private static void Main(string[] args)
    {
        if (Vec162TopKReportCommand.TryRun(args))
        {
            return;
        }

        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
    }
}
