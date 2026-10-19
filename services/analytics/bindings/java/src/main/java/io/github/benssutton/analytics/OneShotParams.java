package io.github.benssutton.analytics;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Keyword parameters of the one-shot recommender; {@link #defaults()} matches the Python
 * {@code analytics.recommend.OneShotRecommender}. The native constructor validates the other parameters; {@code topK} is validated here.
 *
 * @param topK entries per {@code top_k} / {@code inner_top_k} map (0: null cells; empty: every ranked value)
 * @param idColumns columns whose values identify an exemplar row ({@code exemplar_id} /
 *                      {@code null_exemplar_id} become structs of them; empty: Null-typed)
 * @param zstdLevels   ZSTD levels tried for the ClickHouse codec chain and the IPC buffer compression
 *                      (non-empty, distinct, within ZSTD's range; default 1, 3, 9)
 * @param zstdLevelTolerance a higher level wins only if it is smaller by more than this fraction (default 0.01)
 * @param zstdMinSaving ZSTD replaces LZ4 only if it saves at least this fraction (default 0.10)
 * @param gcdMinSaving  the GCD codec is chosen only if it saves at least this fraction (default 0.01)
 * (spec 2026-10-19-clickhouse-recommendations-design.md §4.3; each fraction in [0, 1))
 */
public record OneShotParams(long categoricalThreshold, int zstdLevel, long seed, List<BooleanPair> booleanPairs,
                            OptionalLong topK, List<String> idColumns,
                            List<Integer> zstdLevels, double zstdLevelTolerance, double zstdMinSaving,
                            double gcdMinSaving) {

    public OneShotParams {
        booleanPairs = List.copyOf(booleanPairs);
        Objects.requireNonNull(topK, "topK");
        idColumns = List.copyOf(Objects.requireNonNull(idColumns, "idColumns"));
        zstdLevels = List.copyOf(Objects.requireNonNull(zstdLevels, "zstdLevels"));
        if (topK.isPresent() && topK.getAsLong() < 0) {
            throw new IllegalArgumentException("topK must be non-negative, got " + topK.getAsLong());
        }
    }

    /** As the canonical constructor, with the default compression keywords. */
    public OneShotParams(long categoricalThreshold, int zstdLevel, long seed, List<BooleanPair> booleanPairs,
                         OptionalLong topK, List<String> idColumns) {
        this(categoricalThreshold, zstdLevel, seed, booleanPairs, topK, idColumns, List.of(1, 3, 9), 0.01, 0.10, 0.01);
    }

    /** As the canonical constructor, with no id columns. */
    public OneShotParams(long categoricalThreshold, int zstdLevel, long seed, List<BooleanPair> booleanPairs,
                         OptionalLong topK) {
        this(categoricalThreshold, zstdLevel, seed, booleanPairs, topK, List.of());
    }

    /** As the canonical constructor, with {@code topK} 256 and no id columns. */
    public OneShotParams(long categoricalThreshold, int zstdLevel, long seed, List<BooleanPair> booleanPairs) {
        this(categoricalThreshold, zstdLevel, seed, booleanPairs, OptionalLong.of(256), List.of());
    }

    /** Threshold 10 000, ZSTD 1, seed 0, ("true", "false"), top 256, levels 1, 3, 9, 0.01, 0.10, 0.01. */
    public static OneShotParams defaults() {
        return new OneShotParams(10_000, 1, 0, List.of(new BooleanPair("true", "false")));
    }

    /** {@code topK} as the C ABI's {@code uint64_t top_k}: -1 is {@code UINT64_MAX}, every value. */
    long nativeTopK() {
        return topK.orElse(-1L);
    }
}
