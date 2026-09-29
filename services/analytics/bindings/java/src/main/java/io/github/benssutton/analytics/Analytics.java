package io.github.benssutton.analytics;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
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

    private static final int INVALID_INPUT = 1;

    private static final MethodHandle DESCRIBE_AND_RECOMMEND;
    private static final MethodHandle FREE_ERROR;

    static {
        SymbolLookup library = SymbolLookup.libraryLookup(libraryPath(), Arena.global());
        Linker linker = Linker.nativeLinker();
        DESCRIBE_AND_RECOMMEND = linker.downcallHandle(
            library.findOrThrow("analytics_describe_and_recommend"),
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
        FREE_ERROR = linker.downcallHandle(
            library.findOrThrow("analytics_free_error"), FunctionDescriptor.ofVoid(ADDRESS));
    }

    private Analytics() {}

    private static Path libraryPath() {
        String dir = System.getProperty("analytics.library.dir");
        if (dir == null) {
            throw new IllegalStateException("system property analytics.library.dir is not set");
        }
        Path path = Path.of(dir).resolve(System.mapLibraryName("analytics")).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException(path + " not found: in services/analytics run "
                + "`cargo build --release --no-default-features --target-dir target/capi`");
        }
        return path;
    }

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
            MemorySegment trues = cStrings(arena, pairs.stream().map(Params.BooleanPair::trueValue).toList());
            MemorySegment falses = cStrings(arena, pairs.stream().map(Params.BooleanPair::falseValue).toList());
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
                throw failure(code, error.get(ADDRESS, 0));
            }
            return Data.importArrayStream(allocator, out);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    /** The exception for a non-zero return code; frees the native message. */
    private static RuntimeException failure(int code, MemorySegment message) throws Throwable {
        String text = message.address() == 0
            ? "analytics error code " + code
            : message.reinterpret(Long.MAX_VALUE).getString(0);
        FREE_ERROR.invokeExact(message);
        return code == INVALID_INPUT ? new IllegalArgumentException(text) : new RuntimeException(text);
    }

    /** A native array of NUL-terminated UTF-8 strings, or NULL when empty. */
    private static MemorySegment cStrings(Arena arena, List<String> strings) {
        if (strings.isEmpty()) {
            return MemorySegment.NULL;
        }
        MemorySegment array = arena.allocate(ADDRESS, strings.size());
        for (int i = 0; i < strings.size(); i++) {
            array.setAtIndex(ADDRESS, i, arena.allocateFrom(strings.get(i)));
        }
        return array;
    }
}
