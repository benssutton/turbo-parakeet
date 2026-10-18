package io.github.benssutton.analytics;

/**
 * How the streaming recommender estimates a column's entropy once its distinct values no
 * longer fit the exact phase.
 */
public enum EntropyEstimator {
    /** The bottom-k distinct sample: no extra per-row cost; weak on skewed columns. */
    SAMPLE,
    /** An AMS sketch fed every row: about 4 096 estimators per column. */
    AMS;

    /** The C ABI's {@code uint8_t entropy_estimator}. */
    byte nativeValue() {
        return (byte) ordinal();
    }
}
