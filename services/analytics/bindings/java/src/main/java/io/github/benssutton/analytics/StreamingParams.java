package io.github.benssutton.analytics;

import java.util.List;

/**
 * Keyword parameters of the streaming recommender; {@link #defaults()} matches the Python
 * {@code analytics.recommend.StreamingRecommender}. Validated by the native constructor.
 *
 * @param reservoirRows rows of contiguous blocks sampled for ZSTD sizes and the cross-check
 *                      (0: no sample, ZSTD sizes null); 0 or at least {@code blockRows}
 * @param blockRows     rows per sampled block (at least 1)
 * @param categoricalThreshold the dictionary gate; also sizes each level's distinct sample,
 *                      k = max(categoricalThreshold, 1000), ≈ 50 bytes per sampled value
 */
public record StreamingParams(long reservoirRows, long blockRows, long categoricalThreshold,
                              int zstdLevel, long seed, List<BooleanPair> booleanPairs) {

    public StreamingParams {
        booleanPairs = List.copyOf(booleanPairs);
    }

    /** 524 288 reservoir rows in blocks of 65 536, threshold 10 000, ZSTD 1, seed 0, ("true", "false"). */
    public static StreamingParams defaults() {
        return new StreamingParams(524_288, 65_536, 10_000, 1, 0,
            List.of(new BooleanPair("true", "false")));
    }
}
