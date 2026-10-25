package io.github.benssutton.analytics;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Keyword parameters of the streaming recommender; {@link #defaults()} matches the Python
 * {@code turbo_parakeet.recommenders.StreamingRecommender}. The native constructor validates the other parameters; {@code topK} is validated here.
 *
 * @param reservoirRows rows of contiguous blocks sampled for ZSTD sizes and the cross-check
 *                      (0: no sample, ZSTD sizes null); 0 or at least {@code blockRows}
 * @param blockRows     rows per sampled block (at least 1)
 * @param categoricalThreshold the dictionary gate; also sizes each level's distinct sample,
 *                      k = max(categoricalThreshold, 1000), ≈ 65 bytes per sampled value
 * @param topK          entries per {@code top_k} / {@code inner_top_k} map (0: null cells; empty: every ranked value)
 * @param idColumns     columns whose values identify an exemplar row ({@code exemplar_id} /
 *                      {@code null_exemplar_id} become structs of them; empty: Null-typed)
 * @param entropyEstimator the entropy estimate once the exact phase overflows
 * @param zstdLevels   ZSTD levels tried for the ClickHouse codec chain and the IPC buffer compression
 *                      (non-empty, distinct, each in 1..22; default 1, 3, 9)
 * @param zstdLevelTolerance a higher level wins only if it is smaller by more than this fraction (default 0.01)
 * @param zstdMinSaving ZSTD replaces LZ4 only if it saves at least this fraction (default 0.10)
 * @param transformMinSaving  a transform codec (GCD, Delta, DoubleDelta) is chosen only if it saves at least this fraction over no transform (default 0.01)
 * (each fraction in [0, 1))
 * @param skipGranularities GRANULARITY values tried for the skip indexes (non-empty; default 1, 2, 4, 8)
 * @param setMaxSizes   {@code set(max_size)} values tried (non-empty; default 125, 250)
 * @param bloomFprs     {@code bloom_filter} false-positive rates tried (non-empty, each in (0, 1); default 0.025)
 * @param minSkip       a skip index must let at least this fraction of point queries skip (default 0.3)
 * @param minGain       and gain at least this much over a random row order (default 0.1)
 * @param maxOverflow   a set index may overflow its max_size in at most this fraction of windows (default 0.1)
 * (fractions in [0, 1])
 */
public record StreamingParams(long reservoirRows, long blockRows, long categoricalThreshold,
                              int zstdLevel, long seed, List<BooleanPair> booleanPairs, OptionalLong topK,
                              List<String> idColumns, EntropyEstimator entropyEstimator,
                              List<Integer> zstdLevels, double zstdLevelTolerance, double zstdMinSaving,
                              double transformMinSaving,
                              List<Integer> skipGranularities, List<Long> setMaxSizes, List<Double> bloomFprs,
                              double minSkip, double minGain, double maxOverflow) {

    public StreamingParams {
        booleanPairs = List.copyOf(booleanPairs);
        Objects.requireNonNull(topK, "topK");
        idColumns = List.copyOf(Objects.requireNonNull(idColumns, "idColumns"));
        Objects.requireNonNull(entropyEstimator, "entropyEstimator");
        zstdLevels = List.copyOf(Objects.requireNonNull(zstdLevels, "zstdLevels"));
        skipGranularities = List.copyOf(Objects.requireNonNull(skipGranularities, "skipGranularities"));
        setMaxSizes = List.copyOf(Objects.requireNonNull(setMaxSizes, "setMaxSizes"));
        bloomFprs = List.copyOf(Objects.requireNonNull(bloomFprs, "bloomFprs"));
        if (topK.isPresent() && topK.getAsLong() < 0) {
            throw new IllegalArgumentException("topK must be non-negative, got " + topK.getAsLong());
        }
    }

    /** As the canonical constructor, with the default skip-index keywords. */
    public StreamingParams(long reservoirRows, long blockRows, long categoricalThreshold,
                           int zstdLevel, long seed, List<BooleanPair> booleanPairs, OptionalLong topK,
                           List<String> idColumns, EntropyEstimator entropyEstimator,
                           List<Integer> zstdLevels, double zstdLevelTolerance, double zstdMinSaving,
                           double transformMinSaving) {
        this(reservoirRows, blockRows, categoricalThreshold, zstdLevel, seed, booleanPairs, topK, idColumns,
            entropyEstimator, zstdLevels, zstdLevelTolerance, zstdMinSaving, transformMinSaving, List.of(1, 2, 4, 8), List.of(125L, 250L), List.of(0.025), 0.3, 0.1, 0.1);
    }

    /** As the canonical constructor, with the default compression keywords. */
    public StreamingParams(long reservoirRows, long blockRows, long categoricalThreshold,
                           int zstdLevel, long seed, List<BooleanPair> booleanPairs, OptionalLong topK,
                           List<String> idColumns, EntropyEstimator entropyEstimator) {
        this(reservoirRows, blockRows, categoricalThreshold, zstdLevel, seed, booleanPairs, topK, idColumns,
            entropyEstimator, List.of(1, 3, 9), 0.01, 0.10, 0.01);
    }

    /** As the canonical constructor, with no id columns and the {@code AMS} estimator. */
    public StreamingParams(long reservoirRows, long blockRows, long categoricalThreshold,
                           int zstdLevel, long seed, List<BooleanPair> booleanPairs, OptionalLong topK) {
        this(reservoirRows, blockRows, categoricalThreshold, zstdLevel, seed, booleanPairs, topK,
            List.of(), EntropyEstimator.AMS);
    }

    /** As the canonical constructor, with {@code topK} 256, no id columns and the {@code AMS} estimator. */
    public StreamingParams(long reservoirRows, long blockRows, long categoricalThreshold,
                           int zstdLevel, long seed, List<BooleanPair> booleanPairs) {
        this(reservoirRows, blockRows, categoricalThreshold, zstdLevel, seed, booleanPairs, OptionalLong.of(256),
            List.of(), EntropyEstimator.AMS);
    }

    /** 524 288 reservoir rows in blocks of 65 536, threshold 10 000, ZSTD 1, seed 0, ("true", "false"), top 256, no id columns, the {@code AMS} entropy estimator. */
    public static StreamingParams defaults() {
        return new StreamingParams(524_288, 65_536, 10_000, 1, 0,
            List.of(new BooleanPair("true", "false")));
    }

    /** {@code topK} as the C ABI's {@code uint64_t top_k}: -1 is {@code UINT64_MAX}, every value. */
    long nativeTopK() {
        return topK.orElse(-1L);
    }
}
