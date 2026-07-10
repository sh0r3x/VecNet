namespace VecNet;

internal readonly record struct HnswBasePlusExactDeltaCheckpointResult(
    HnswBasePlusExactDeltaCheckpointStatus Status,
    long Generation,
    int RebuiltBaseVectorCount,
    int LiveVectorCount,
    int BasePhysicalVectorCount,
    int BaseLiveVectorCount,
    int DeltaPhysicalVectorCount,
    int DeltaLiveVectorCount,
    int BaseTombstoneCount,
    int DeltaTombstoneCount,
    int TombstoneCount,
    int DeletedReservedIdCount,
    int FoldedDeltaVectorCount,
    int FoldedBaseTombstoneCount,
    int FoldedDeltaTombstoneCount);
