namespace VecNet;

internal readonly record struct HnswBasePlusExactDeltaCheckpointDiagnostics(
    HnswBasePlusExactDeltaCheckpointPhaseDiagnostics LiveSnapshot,
    HnswBasePlusExactDeltaCheckpointPhaseDiagnostics RebuildBuild,
    HnswBasePlusExactDeltaCheckpointPhaseDiagnostics Save,
    HnswBasePlusExactDeltaCheckpointPhaseDiagnostics OpenValidation,
    HnswBasePlusExactDeltaCheckpointPhaseDiagnostics Publication)
{
    internal static HnswBasePlusExactDeltaCheckpointDiagnostics None { get; } =
        new(
            HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted,
            HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted,
            HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted,
            HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted,
            HnswBasePlusExactDeltaCheckpointPhaseDiagnostics.NotExecuted);
}
