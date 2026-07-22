namespace VecNet.BenchmarkRunner;

public static class ExactGeneratedPublicEvidencePolicy
{
    public const string PolicyName = "VEC-215 exact-generated public evidence validation policy";
    public const string PolicyVersion = "0.1";
    public const double AcceptedRecallFloor = 0.999;
    public const string NearTieTolerancePolicy =
        "Near-tie/order-only acceptance uses the same scalar-reference distance tolerance as generated exact distance validation: squared L2 max(2e-4, (8 * dimension / 16777216) * max(1, abs(scalarDistance))); other metrics use 1e-5 * max(1, abs(scalarDistance)).";

    public static ExactGeneratedPublicEvidenceValidationInfo Evaluate(
        TruthSet truth,
        SearchResult[][] actual,
        GeneratedDataset dataset,
        VectorMetric metric,
        int topK,
        int dimension,
        ResultComparison strictComparison)
    {
        ArgumentNullException.ThrowIfNull(dataset);

        return Evaluate(
            truth,
            actual,
            metric,
            topK,
            dimension,
            strictComparison,
            (queryRow, id) =>
            {
                if (id > int.MaxValue || id >= (ulong)dataset.VectorCount)
                {
                    return null;
                }

                return ScalarGroundTruth.CalculateDistance(
                    dataset.GetQuery(queryRow),
                    dataset.GetVector((int)id),
                    metric);
            });
    }

    internal static ExactGeneratedPublicEvidenceValidationInfo Evaluate(
        TruthSet truth,
        SearchResult[][] actual,
        VectorMetric metric,
        int topK,
        int dimension,
        ResultComparison strictComparison,
        Func<int, ulong, float?> scalarDistanceResolver)
    {
        ArgumentNullException.ThrowIfNull(truth);
        ArgumentNullException.ThrowIfNull(actual);
        ArgumentNullException.ThrowIfNull(strictComparison);
        ArgumentNullException.ThrowIfNull(scalarDistanceResolver);

        if (truth.Results.Length != actual.Length)
        {
            throw new ArgumentException("Truth and actual result query counts differ.", nameof(actual));
        }

        int missingResultCount = 0;
        int duplicateResultCount = 0;
        int distanceMismatchCount = 0;
        int wrongIdAwayFromNearTieCount = 0;
        int boundaryNearTieMismatchCount = 0;
        int orderMismatchCount = 0;
        List<string> diagnostics = [];

        for (int queryRow = 0; queryRow < truth.Results.Length; queryRow++)
        {
            TruthItem[] expected = truth.Results[queryRow];
            SearchResult[] returned = actual[queryRow];
            int denominator = Math.Min(topK, expected.Length);
            if (denominator == 0)
            {
                continue;
            }

            if (returned.Length < denominator)
            {
                missingResultCount += denominator - returned.Length;
            }

            var expectedIds = new HashSet<ulong>();
            for (int i = 0; i < denominator; i++)
            {
                expectedIds.Add(expected[i].Id);
            }

            var returnedIds = new HashSet<ulong>();
            var returnedTop = returned.Take(denominator).ToArray();
            foreach (SearchResult result in returnedTop)
            {
                if (!returnedIds.Add(result.Id))
                {
                    duplicateResultCount++;
                }

                float? scalarDistance = scalarDistanceResolver(queryRow, result.Id);
                if (scalarDistance is null ||
                    !ResultComparer.DistanceMatches(scalarDistance.Value, result.Distance, dimension, metric))
                {
                    distanceMismatchCount++;
                }
            }

            float boundaryDistance = expected[denominator - 1].Distance;
            foreach (TruthItem expectedItem in expected.Take(denominator))
            {
                if (returnedIds.Contains(expectedItem.Id))
                {
                    continue;
                }

                if (IsNearTie(boundaryDistance, expectedItem.Distance, dimension, metric))
                {
                    boundaryNearTieMismatchCount++;
                }
                else
                {
                    wrongIdAwayFromNearTieCount++;
                }
            }

            foreach (SearchResult result in returnedTop)
            {
                if (expectedIds.Contains(result.Id))
                {
                    continue;
                }

                float? scalarDistance = scalarDistanceResolver(queryRow, result.Id);
                if (scalarDistance is not null &&
                    IsNearTie(boundaryDistance, scalarDistance.Value, dimension, metric))
                {
                    boundaryNearTieMismatchCount++;
                }
                else
                {
                    wrongIdAwayFromNearTieCount++;
                }
            }

            for (int i = 0; i < Math.Min(denominator, returnedTop.Length); i++)
            {
                if (returnedTop[i].Id == expected[i].Id)
                {
                    continue;
                }

                orderMismatchCount++;
                float? scalarDistance = scalarDistanceResolver(queryRow, returnedTop[i].Id);
                if (scalarDistance is null ||
                    !IsNearTie(expected[i].Distance, scalarDistance.Value, dimension, metric))
                {
                    wrongIdAwayFromNearTieCount++;
                }
            }
        }

        bool strictPerfect =
            strictComparison.RecallAtK == 1 &&
            strictComparison.OrderedAgreement == 1 &&
            strictComparison.DistanceToleranceStatus == "passed" &&
            strictComparison.DistanceMismatchCount == 0 &&
            strictComparison.MissingResultCount == 0 &&
            missingResultCount == 0 &&
            duplicateResultCount == 0 &&
            distanceMismatchCount == 0;

        bool nearTieOrderOnly =
            !strictPerfect &&
            strictComparison.RecallAtK >= AcceptedRecallFloor &&
            strictComparison.DistanceToleranceStatus == "passed" &&
            strictComparison.MissingResultCount == 0 &&
            missingResultCount == 0 &&
            duplicateResultCount == 0 &&
            distanceMismatchCount == 0 &&
            wrongIdAwayFromNearTieCount == 0 &&
            (orderMismatchCount > 0 || boundaryNearTieMismatchCount > 0);

        string status = strictPerfect
            ? "passed-strict"
            : nearTieOrderOnly
                ? "accepted-near-tie-order-only"
                : "failed";
        string classification = strictPerfect
            ? "strict-perfect-order"
            : nearTieOrderOnly
                ? "deterministic-near-tie-order-only"
                : "failed-public-evidence-validation";

        AddDiagnostics(
            diagnostics,
            strictComparison,
            missingResultCount,
            duplicateResultCount,
            distanceMismatchCount,
            wrongIdAwayFromNearTieCount,
            boundaryNearTieMismatchCount,
            orderMismatchCount,
            strictPerfect,
            nearTieOrderOnly);

        return new ExactGeneratedPublicEvidenceValidationInfo(
            PolicyName,
            PolicyVersion,
            status,
            strictPerfect || nearTieOrderOnly,
            classification,
            strictComparison.RecallAtK,
            strictComparison.OrderedAgreement,
            AcceptedRecallFloor,
            distanceMismatchCount == 0 && missingResultCount == 0 ? "passed" : "failed",
            distanceMismatchCount,
            missingResultCount,
            duplicateResultCount,
            wrongIdAwayFromNearTieCount,
            boundaryNearTieMismatchCount,
            orderMismatchCount,
            NearTieTolerancePolicy,
            CreateExplanation(strictPerfect, nearTieOrderOnly),
            diagnostics.ToArray());
    }

    private static bool IsNearTie(float left, float right, int dimension, VectorMetric metric)
    {
        float tolerance = ResultComparer.CalculateDistanceTolerance(dimension, metric, left);
        return MathF.Abs(left - right) <= tolerance;
    }

    private static void AddDiagnostics(
        List<string> diagnostics,
        ResultComparison strictComparison,
        int missingResultCount,
        int duplicateResultCount,
        int distanceMismatchCount,
        int wrongIdAwayFromNearTieCount,
        int boundaryNearTieMismatchCount,
        int orderMismatchCount,
        bool strictPerfect,
        bool nearTieOrderOnly)
    {
        diagnostics.Add($"strictRecallAtK={strictComparison.RecallAtK}");
        diagnostics.Add($"strictOrderedAgreement={strictComparison.OrderedAgreement}");
        diagnostics.Add($"acceptedRecallFloor={AcceptedRecallFloor}");
        diagnostics.Add($"distanceMismatchCount={distanceMismatchCount}");
        diagnostics.Add($"missingResultCount={missingResultCount}");
        diagnostics.Add($"duplicateResultCount={duplicateResultCount}");
        diagnostics.Add($"wrongIdAwayFromNearTieCount={wrongIdAwayFromNearTieCount}");
        diagnostics.Add($"boundaryNearTieMismatchCount={boundaryNearTieMismatchCount}");
        diagnostics.Add($"orderMismatchCount={orderMismatchCount}");
        diagnostics.Add(strictPerfect
            ? "strict exact validation passed with perfect ordered agreement."
            : nearTieOrderOnly
                ? "ordered agreement below 1.0 is diagnostic because all differences are deterministic near-tie/order-only differences under the stated tolerance policy."
                : "public exact-generated evidence is not acceptable because one or more strict or near-tie policy conditions failed.");
    }

    private static string CreateExplanation(bool strictPerfect, bool nearTieOrderOnly) =>
        strictPerfect
            ? "Strict exact-generated validation passed with perfect recall, ordering and distance agreement."
            : nearTieOrderOnly
                ? "Recall is perfect or within the accepted near-boundary floor, returned distances match scalar-reference distances, no results are missing, and every order/set difference is within the deterministic near-tie boundary between scalar-reference truth and production exact search."
                : "The report is not acceptable as public exact-generated evidence because it has missing results, distance mismatches, duplicate results, recall below the accepted floor or wrong IDs/order away from near-tie boundaries.";
}
