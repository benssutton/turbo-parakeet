package io.github.benssutton.analytics;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

/**
 * Recommend's dtype recommendations from batches added over time; all state stays in Rust
 * (services/analytics/src/recommenders/streaming/, through the C ABI's {@code analytics_streaming_recommender_*}).
 * Spec: docs/superpowers/specs/2026-09-29-streaming-recommender-design.md.
 *
 * <p>{@link #add} any number of times — columns may appear, disappear (their rows count as
 * null) or start as the Null type; any other type change is an {@link IllegalArgumentException}.
 * {@link #result} at any point returns one row per column and keeps the state.
 *
 * <p>Memory per eligible level (a column, or a list's inner values), every dtype: ≈ 16 KB of
 * HyperLogLog plus a distinct sample of up to ≈ 65 bytes × k, k = max(categoricalThreshold,
 * 1000) — ≈ 0.65 MB at the default 10 000, so ≈ 0.65 GB for 1 000 high-cardinality columns.
 * Text levels also keep their distinct values' text while exact (k × mean value length).
 */
public final class StreamingRecommender extends NativeRecommender {

    private static final Functions FUNCTIONS = Functions.of("analytics_streaming_recommender");
    private static final MethodHandle NEW = Native.handle("analytics_streaming_recommender_new",
        FunctionDescriptor.of(JAVA_INT,
            JAVA_LONG,  // uint64_t reservoir_rows
            JAVA_LONG,  // uint64_t block_rows
            JAVA_LONG,  // uint64_t categorical_threshold
            JAVA_LONG,  // uint64_t top_k
            JAVA_INT,   // int32_t zstd_level
            JAVA_LONG,  // uint64_t seed
            ADDRESS,    // const char *const *bool_true
            ADDRESS,    // const char *const *bool_false
            JAVA_LONG,  // size_t n_bool_pairs
            ADDRESS,    // void **out
            ADDRESS));  // char **error

    /** @throws IllegalArgumentException for invalid parameters (e.g. {@code blockRows} 0) */
    public StreamingRecommender(StreamingParams params) {
        super(FUNCTIONS, create(params));
    }

    private static MemorySegment create(StreamingParams params) {
        return Native.call(() -> {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment[] pairs = booleanPairs(arena, params.booleanPairs());
                MemorySegment out = arena.allocate(ADDRESS);
                MemorySegment error = arena.allocate(ADDRESS);  // zero-initialised: null
                int code = (int) NEW.invokeExact(
                    params.reservoirRows(),
                    params.blockRows(),
                    params.categoricalThreshold(),
                    params.nativeTopK(),
                    params.zstdLevel(),
                    params.seed(),
                    pairs[0],
                    pairs[1],
                    (long) params.booleanPairs().size(),
                    out,
                    error);
                if (code != 0) {
                    throw Native.failure(code, error.get(ADDRESS, 0));
                }
                return out.get(ADDRESS, 0);
            }
        });
    }
}
