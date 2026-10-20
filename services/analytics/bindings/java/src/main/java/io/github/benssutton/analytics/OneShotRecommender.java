package io.github.benssutton.analytics;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

/**
 * The narrowest value-preserving Arrow type per column of one frame, from exact statistics,
 * each candidate cast, verified on every row and measured (services/analytics/src/recommenders/oneshot.rs,
 * through the C ABI's {@code analytics_oneshot_recommender_*}). Spec:
 * docs/superpowers/specs/2026-10-04-oneshot-recommender-design.md.
 *
 * <p>{@link #add} once (its batches are concatenated; a second call is an
 * {@link IllegalArgumentException}); {@link #result} any number of times returns the same
 * table: the streaming recommender's columns without {@code first_row},
 * {@code n_sampled_rows} and {@code n_sampled_blocks}.
 */
public final class OneShotRecommender extends NativeRecommender {

    private static final Functions FUNCTIONS = Functions.of("analytics_oneshot_recommender");
    private static final MethodHandle NEW = Native.handle("analytics_oneshot_recommender_new",
        FunctionDescriptor.of(JAVA_INT,
            JAVA_LONG,  // uint64_t categorical_threshold
            JAVA_LONG,  // uint64_t top_k
            JAVA_INT,   // int32_t zstd_level
            JAVA_LONG,  // uint64_t seed
            ADDRESS,    // const char *const *bool_true
            ADDRESS,    // const char *const *bool_false
            JAVA_LONG,  // size_t n_bool_pairs
            ADDRESS,    // const char *const *id_columns
            JAVA_LONG,  // size_t n_id_columns
            ADDRESS,    // const int32_t *zstd_levels
            JAVA_LONG,  // size_t n_zstd_levels
            JAVA_DOUBLE,  // double zstd_level_tolerance
            JAVA_DOUBLE,  // double zstd_min_saving
            JAVA_DOUBLE,  // double gcd_min_saving
            ADDRESS,    // const uint32_t *skip_granularities
            JAVA_LONG,  // size_t n_skip_granularities
            ADDRESS,    // const uint64_t *set_max_sizes
            JAVA_LONG,  // size_t n_set_max_sizes
            ADDRESS,    // const double *bloom_fprs
            JAVA_LONG,  // size_t n_bloom_fprs
            JAVA_DOUBLE,  // double min_skip
            JAVA_DOUBLE,  // double min_gain
            JAVA_DOUBLE,  // double max_overflow
            ADDRESS,    // void **out
            ADDRESS));  // char **error

    /** @throws IllegalArgumentException for invalid parameters (e.g. a ZSTD level out of range) */
    public OneShotRecommender(OneShotParams params) {
        super(FUNCTIONS, create(params));
    }

    private static MemorySegment create(OneShotParams params) {
        return Native.call(() -> {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment[] pairs = booleanPairs(arena, params.booleanPairs());
                MemorySegment out = arena.allocate(ADDRESS);
                MemorySegment error = arena.allocate(ADDRESS);  // zero-initialised: null
                int code = (int) NEW.invokeExact(
                    params.categoricalThreshold(),
                    params.nativeTopK(),
                    params.zstdLevel(),
                    params.seed(),
                    pairs[0],
                    pairs[1],
                    (long) params.booleanPairs().size(),
                    Native.cStrings(arena, params.idColumns()),
                    (long) params.idColumns().size(),
                    arena.allocateFrom(JAVA_INT, params.zstdLevels().stream().mapToInt(Integer::intValue).toArray()),
                    (long) params.zstdLevels().size(),
                    params.zstdLevelTolerance(),
                    params.zstdMinSaving(),
                    params.gcdMinSaving(),
                    arena.allocateFrom(JAVA_INT, params.skipGranularities().stream().mapToInt(Integer::intValue).toArray()),
                    (long) params.skipGranularities().size(),
                    arena.allocateFrom(JAVA_LONG, params.setMaxSizes().stream().mapToLong(Long::longValue).toArray()),
                    (long) params.setMaxSizes().size(),
                    arena.allocateFrom(JAVA_DOUBLE, params.bloomFprs().stream().mapToDouble(Double::doubleValue).toArray()),
                    (long) params.bloomFprs().size(),
                    params.minSkip(),
                    params.minGain(),
                    params.maxOverflow(),
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
