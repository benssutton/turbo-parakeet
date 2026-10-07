package io.github.benssutton.analytics;

import java.util.List;
import java.util.OptionalLong;

/**
 * Keyword parameters of the one-shot recommender; {@link #defaults()} matches the Python
 * {@code analytics.recommend.OneShotRecommender}. Validated by the native constructor.
 *
 * @param topK entries per {@code top_k} / {@code inner_top_k} map (0 none; empty: every ranked value)
 */
public record OneShotParams(long categoricalThreshold, int zstdLevel, long seed, List<BooleanPair> booleanPairs,
                            OptionalLong topK) {

    public OneShotParams {
        booleanPairs = List.copyOf(booleanPairs);
        if (topK.isPresent() && topK.getAsLong() < 0) {
            throw new IllegalArgumentException("topK must be non-negative, got " + topK.getAsLong());
        }
    }

    /** As the canonical constructor, with {@code topK} 256. */
    public OneShotParams(long categoricalThreshold, int zstdLevel, long seed, List<BooleanPair> booleanPairs) {
        this(categoricalThreshold, zstdLevel, seed, booleanPairs, OptionalLong.of(256));
    }

    /** Threshold 10 000, ZSTD 1, seed 0, ("true", "false"), top 256. */
    public static OneShotParams defaults() {
        return new OneShotParams(10_000, 1, 0, List.of(new BooleanPair("true", "false")));
    }

    /** {@code topK} as the C ABI's {@code uint64_t top_k}: -1 is {@code UINT64_MAX}, every value. */
    long nativeTopK() {
        return topK.orElse(-1L);
    }
}
