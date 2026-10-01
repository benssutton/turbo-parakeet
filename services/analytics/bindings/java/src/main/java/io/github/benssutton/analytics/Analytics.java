package io.github.benssutton.analytics;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.List;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * Java binding over the analytics C ABI (services/analytics/src/capi.rs). Tables cross as
 * Arrow C Streams, without copying. The native library is found in the directory named by
 * the system property {@code analytics.library.dir}.
 */
public final class Analytics {

    private static final MethodHandle DESCRIBE_AND_RECOMMEND = Native.handle(
        "analytics_describe_and_recommend",
        FunctionDescriptor.of(JAVA_INT,
            ADDRESS,    // ArrowArrayStream *input (consumed)
            JAVA_LONG,  // uint64_t seed
            JAVA_INT,   // int32_t zstd_level
            JAVA_LONG,  // int64_t population_rows (< 0 = none)
            JAVA_LONG,  // uint64_t categorical_threshold
            ADDRESS,    // const char *const *bool_true
            ADDRESS,    // const char *const *bool_false
            JAVA_LONG,  // size_t n_bool_pairs
            ADDRESS,    // ArrowArrayStream *output
            ADDRESS));  // char **error

    private Analytics() {}

    /**
     * Describe's table, the size columns and the {@code rec_*} columns, one row per column of
     * {@code input}. {@code input} is consumed. The returned reader holds one batch; the
     * caller closes it.
     *
     * @throws IllegalArgumentException for invalid input (e.g. duplicate column names)
     * @throws RuntimeException for a failure inside the Rust kernels
     */
    public static ArrowReader describeAndRecommend(ArrowReader input, Params params, BufferAllocator allocator) {
        try (ArrowArrayStream in = ArrowArrayStream.allocateNew(allocator);
             ArrowArrayStream out = ArrowArrayStream.allocateNew(allocator);
             Arena arena = Arena.ofConfined()) {
            Data.exportArrayStream(allocator, input, in);
            List<Params.BooleanPair> pairs = params.booleanPairs();
            MemorySegment trues = Native.cStrings(arena, pairs.stream().map(Params.BooleanPair::trueValue).toList());
            MemorySegment falses = Native.cStrings(arena, pairs.stream().map(Params.BooleanPair::falseValue).toList());
            MemorySegment error = arena.allocate(ADDRESS);  // zero-initialised: null
            int code = (int) DESCRIBE_AND_RECOMMEND.invokeExact(
                MemorySegment.ofAddress(in.memoryAddress()),
                params.seed(),
                params.zstdLevel(),
                params.populationRows().orElse(-1),
                params.categoricalThreshold(),
                trues,
                falses,
                (long) pairs.size(),
                MemorySegment.ofAddress(out.memoryAddress()),
                error);
            if (code != 0) {
                throw Native.failure(code, error.get(ADDRESS, 0));
            }
            return Data.importArrayStream(allocator, out);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }
}
