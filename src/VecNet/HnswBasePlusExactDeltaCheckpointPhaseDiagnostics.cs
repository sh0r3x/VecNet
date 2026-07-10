namespace VecNet;

internal readonly record struct HnswBasePlusExactDeltaCheckpointPhaseDiagnostics(
    HnswBasePlusExactDeltaCheckpointPhaseStatus Status,
    long ElapsedTicks,
    long ManagedAllocatedBytes)
{
    internal static HnswBasePlusExactDeltaCheckpointPhaseDiagnostics NotExecuted { get; } =
        new(HnswBasePlusExactDeltaCheckpointPhaseStatus.NotExecuted, 0, 0);

    internal static HnswBasePlusExactDeltaCheckpointPhaseDiagnostics Measured(
        long elapsedTicks,
        long managedAllocatedBytes) =>
        new(HnswBasePlusExactDeltaCheckpointPhaseStatus.Measured, elapsedTicks, managedAllocatedBytes);

    internal static HnswBasePlusExactDeltaCheckpointPhaseDiagnostics Failed(
        long elapsedTicks,
        long managedAllocatedBytes) =>
        new(HnswBasePlusExactDeltaCheckpointPhaseStatus.Failed, elapsedTicks, managedAllocatedBytes);
}
