namespace VecNet;

internal readonly record struct HnswQueueItem(int Ordinal, float Distance);

internal static class HnswPriorityQueues
{
    internal static void PushNearest(
        int[] ordinals,
        float[] distances,
        ulong[] ids,
        ref int count,
        int ordinal,
        float distance)
    {
        if ((uint)count >= (uint)ordinals.Length || (uint)count >= (uint)distances.Length)
        {
            throw new InvalidOperationException("HNSW candidate queue capacity was exceeded.");
        }

        ordinals[count] = ordinal;
        distances[count] = distance;
        SiftUpNearest(ordinals, distances, ids, count);
        count++;
    }

    internal static HnswQueueItem PopNearest(int[] ordinals, float[] distances, ulong[] ids, ref int count)
    {
        if (count <= 0)
        {
            throw new InvalidOperationException("HNSW candidate queue is empty.");
        }

        int ordinal = ordinals[0];
        float distance = distances[0];
        count--;
        if (count > 0)
        {
            ordinals[0] = ordinals[count];
            distances[0] = distances[count];
            SiftDownNearest(ordinals, distances, ids, count, 0);
        }

        return new HnswQueueItem(ordinal, distance);
    }

    internal static bool AddBoundedNearest(
        int[] ordinals,
        float[] distances,
        ulong[] ids,
        ref int count,
        int maxCount,
        int ordinal,
        float distance)
    {
        if (maxCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxCount), "HNSW accepted-result capacity must be positive.");
        }

        if (count < maxCount)
        {
            PushWorst(ordinals, distances, ids, ref count, ordinal, distance);
            return true;
        }

        if (CompareNearest(distance, ids[ordinal], ordinal, distances[0], ids[ordinals[0]], ordinals[0]) >= 0)
        {
            return false;
        }

        ordinals[0] = ordinal;
        distances[0] = distance;
        SiftDownWorst(ordinals, distances, ids, count, 0);
        return true;
    }

    internal static HnswQueueItem PeekWorst(int[] ordinals, float[] distances, int count)
    {
        if (count <= 0)
        {
            throw new InvalidOperationException("HNSW accepted-result queue is empty.");
        }

        return new HnswQueueItem(ordinals[0], distances[0]);
    }

    internal static int CompareNearest(
        float leftDistance,
        ulong leftId,
        int leftOrdinal,
        float rightDistance,
        ulong rightId,
        int rightOrdinal)
    {
        int distanceComparison = leftDistance.CompareTo(rightDistance);
        if (distanceComparison != 0)
        {
            return distanceComparison;
        }

        int idComparison = leftId.CompareTo(rightId);
        return idComparison != 0 ? idComparison : leftOrdinal.CompareTo(rightOrdinal);
    }

    private static void PushWorst(
        int[] ordinals,
        float[] distances,
        ulong[] ids,
        ref int count,
        int ordinal,
        float distance)
    {
        if ((uint)count >= (uint)ordinals.Length || (uint)count >= (uint)distances.Length)
        {
            throw new InvalidOperationException("HNSW accepted-result queue capacity was exceeded.");
        }

        ordinals[count] = ordinal;
        distances[count] = distance;
        SiftUpWorst(ordinals, distances, ids, count);
        count++;
    }

    private static void SiftUpNearest(int[] ordinals, float[] distances, ulong[] ids, int index)
    {
        while (index > 0)
        {
            int parent = (index - 1) >> 1;
            if (CompareNearestAt(ordinals, distances, ids, index, parent) >= 0)
            {
                break;
            }

            Swap(ordinals, distances, index, parent);
            index = parent;
        }
    }

    private static void SiftDownNearest(int[] ordinals, float[] distances, ulong[] ids, int count, int index)
    {
        while (true)
        {
            int left = (index << 1) + 1;
            if (left >= count)
            {
                break;
            }

            int best = left;
            int right = left + 1;
            if (right < count && CompareNearestAt(ordinals, distances, ids, right, left) < 0)
            {
                best = right;
            }

            if (CompareNearestAt(ordinals, distances, ids, best, index) >= 0)
            {
                break;
            }

            Swap(ordinals, distances, index, best);
            index = best;
        }
    }

    private static void SiftUpWorst(int[] ordinals, float[] distances, ulong[] ids, int index)
    {
        while (index > 0)
        {
            int parent = (index - 1) >> 1;
            if (CompareNearestAt(ordinals, distances, ids, index, parent) <= 0)
            {
                break;
            }

            Swap(ordinals, distances, index, parent);
            index = parent;
        }
    }

    private static void SiftDownWorst(int[] ordinals, float[] distances, ulong[] ids, int count, int index)
    {
        while (true)
        {
            int left = (index << 1) + 1;
            if (left >= count)
            {
                break;
            }

            int worst = left;
            int right = left + 1;
            if (right < count && CompareNearestAt(ordinals, distances, ids, right, left) > 0)
            {
                worst = right;
            }

            if (CompareNearestAt(ordinals, distances, ids, worst, index) <= 0)
            {
                break;
            }

            Swap(ordinals, distances, index, worst);
            index = worst;
        }
    }

    private static int CompareNearestAt(int[] ordinals, float[] distances, ulong[] ids, int left, int right) =>
        CompareNearest(
            distances[left],
            ids[ordinals[left]],
            ordinals[left],
            distances[right],
            ids[ordinals[right]],
            ordinals[right]);

    private static void Swap(int[] ordinals, float[] distances, int left, int right)
    {
        (ordinals[left], ordinals[right]) = (ordinals[right], ordinals[left]);
        (distances[left], distances[right]) = (distances[right], distances[left]);
    }
}
