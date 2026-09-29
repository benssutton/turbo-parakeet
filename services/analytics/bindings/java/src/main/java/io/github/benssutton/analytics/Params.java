package io.github.benssutton.analytics;

import java.util.List;
import java.util.OptionalLong;

/**
 * Keyword parameters of {@code describe_and_recommend}; {@link #defaults()} matches the
 * Python technique ({@code analytics.recommend}).
 */
public record Params(long seed, int zstdLevel, OptionalLong populationRows,
                     long categoricalThreshold, List<BooleanPair> booleanPairs) {

    /** Two strings that together mark a string column as boolean, e.g. ("true", "false"). */
    public record BooleanPair(String trueValue, String falseValue) {}

    public Params {
        booleanPairs = List.copyOf(booleanPairs);
    }

    /** seed 0, ZSTD level 1, no population, categorical threshold 10 000, ("true", "false"). */
    public static Params defaults() {
        return new Params(0, 1, OptionalLong.empty(), 10_000, List.of(new BooleanPair("true", "false")));
    }
}
