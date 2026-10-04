package io.github.benssutton.analytics;

/** Two strings that together mark a string column as boolean, e.g. ("true", "false"). */
public record BooleanPair(String trueValue, String falseValue) {}
