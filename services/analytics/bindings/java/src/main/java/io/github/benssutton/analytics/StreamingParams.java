package io.github.benssutton.analytics;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Keyword parameters of the streaming recommender; {@link #defaults()} matches the Python
 * {@code analytics.recommend.StreamingRecommender}. The native constructor validates the other parameters; {@code topK} is validated here.
 *
 * @param reservoirRows rows of contiguous blocks sampled for ZSTD sizes and the cross-check
 *                      (0: no sample, ZSTD sizes null); 0 or at least {@code blockRows}
 * @param blockRows     rows per sampled block (at least 1)
 * @param categoricalThreshold the dictionary gate; also sizes each level's distinct sample,
 *                      k = max(categoricalThreshold, 1000), ≈ 65 bytes per sampled value
 * @param topK          entries per {@code top_k} / {@code inner_top_k} map (0: null cells; empty: every ranked value)
 */
public record StreamingParams(long reservoirRows, long blockRows, long categoricalThreshold,
                              int zstdLevel, long seed, List<BooleanPair> booleanPairs, OptionalLong topK) {

    public StreamingParams {
        booleanPairs = List.copyOf(booleanPairs);
        Objects.requireNonNull(topK, "topK");
        if (topK.isPresent() && topK.getAsLong() < 0) {
            throw new IllegalArgumentException("topK must be non-negative, got " + topK.getAsLong());
        }
    }

    /** As the canonical constructor, with {@code topK} 256. */
    public StreamingParams(long reservoirRows, long blockRows, long categoricalThreshold,
                           int zstdLevel, long seed, List<BooleanPair> booleanPairs) {
        this(reservoirRows, blockRows, categoricalThreshold, zstdLevel, seed, booleanPairs, OptionalLong.of(256));
    }

    /** 524 288 reservoir rows in blocks of 65 536, threshold 10 000, ZSTD 1, seed 0, ("true", "false"), top 256. */
    public static StreamingParams defaults() {
        return new StreamingParams(524_288, 65_536, 10_000, 1, 0,
            List.of(new BooleanPair("true", "false")));
    }

    /** {@code topK} as the C ABI's {@code uint64_t top_k}: -1 is {@code UINT64_MAX}, every value. */
    long nativeTopK() {
        return topK.orElse(-1L);
    }
}
