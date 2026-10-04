package io.github.benssutton.analytics;

import static java.lang.foreign.ValueLayout.ADDRESS;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * The analytics native library (services/analytics/src/capi.rs), shared by the bindings: the
 * library lookup, downcall handles, and the C ABI's error convention (0 ok, 1 invalid input,
 * 2 compute failure, with a message the caller frees).
 */
final class Native {

    static final int INVALID_INPUT = 1;

    private static final SymbolLookup LIBRARY = SymbolLookup.libraryLookup(
        libraryPath(System.getProperty("analytics.library.dir")), Arena.global());
    private static final Linker LINKER = Linker.nativeLinker();
    private static final MethodHandle FREE_ERROR = handle("analytics_free_error", FunctionDescriptor.ofVoid(ADDRESS));

    private Native() {}

    /** The library file in {@code dir} (the value of the system property {@code analytics.library.dir}). */
    static Path libraryPath(String dir) {
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

    /** A native call: the downcall handles throw {@link Throwable}. */
    @FunctionalInterface
    interface Call<T> {
        T run() throws Throwable;
    }

    /** {@link Call} without a result. */
    @FunctionalInterface
    interface VoidCall {
        void run() throws Throwable;
    }

    /** Runs {@code call}; runtime exceptions and errors pass through, anything else is wrapped. */
    static <T> T call(Call<T> call) {
        try {
            return call.run();
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new AnalyticsException(t);
        }
    }

    static void run(VoidCall call) {
        call(() -> {
            call.run();
            return null;
        });
    }

    /** A downcall handle for the library function {@code name}. */
    static MethodHandle handle(String name, FunctionDescriptor descriptor) {
        return LINKER.downcallHandle(LIBRARY.findOrThrow(name), descriptor);
    }

    /** The exception for a non-zero return code; frees the native message. */
    static RuntimeException failure(int code, MemorySegment message) throws Throwable {
        String text = message.address() == 0
            ? "analytics error code " + code
            : message.reinterpret(Long.MAX_VALUE).getString(0);
        FREE_ERROR.invokeExact(message);
        return code == INVALID_INPUT ? new IllegalArgumentException(text) : new AnalyticsException(text);
    }

    /** A native array of NUL-terminated UTF-8 strings, or NULL when empty. */
    static MemorySegment cStrings(Arena arena, List<String> strings) {
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
