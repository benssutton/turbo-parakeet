package io.github.benssutton.analytics;

import java.util.List;

/**
 * Keyword parameters of the one-shot recommender; {@link #defaults()} matches the Python
 * {@code analytics.recommend.OneShotRecommender}. Validated by the native constructor.
 */
public record OneShotParams(long categoricalThreshold, int zstdLevel, long seed, List<BooleanPair> booleanPairs) {

    public OneShotParams {
        booleanPairs = List.copyOf(booleanPairs);
    }

    /** Threshold 10 000, ZSTD 1, seed 0, ("true", "false"). */
    public static OneShotParams defaults() {
        return new OneShotParams(10_000, 1, 0, List.of(new BooleanPair("true", "false")));
    }
}
