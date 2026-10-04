package io.github.benssutton.analytics;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Cleaner;
import java.util.List;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * A recommender whose state lives in Rust behind an opaque handle of the C ABI
 * ({@code <prefix>_new / _add / _result / _free}, services/analytics/src/capi.rs): the
 * lifecycle {@link OneShotRecommender} and {@link StreamingRecommender} share.
 *
 * <p>Thread safety: {@code add} and {@code result} may be called from several threads (the
 * native side serialises them); {@link #close} waits for calls in progress. A producer whose
 * callbacks re-enter this recommender while {@code add} pulls its stream deadlocks.
 */
abstract sealed class NativeRecommender implements AutoCloseable
    permits OneShotRecommender, StreamingRecommender {

    /** The {@code _add}, {@code _result} and {@code _free} functions of one C ABI prefix. */
    record Functions(MethodHandle add, MethodHandle result, MethodHandle free) {
        static Functions of(String prefix) {
            FunctionDescriptor call = FunctionDescriptor.of(JAVA_INT,
                ADDRESS,    // recommender handle
                ADDRESS,    // ArrowArrayStream * (add: consumed input; result: output)
                ADDRESS);   // char **error
            return new Functions(
                Native.handle(prefix + "_add", call),
                Native.handle(prefix + "_result", call),
                Native.handle(prefix + "_free", FunctionDescriptor.ofVoid(ADDRESS)));
        }
    }

    private static final Cleaner CLEANER = Cleaner.create();

    /** Frees the native recommender; run once, by {@link #close} or when unreachable. */
    private record Free(MethodHandle free, MemorySegment handle) implements Runnable {
        @Override
        public void run() {
            // A block body: as an expression lambda, invokeExact would link as returning Object.
            Native.run(() -> {
                free.invokeExact(handle);
            });
        }
    }

    private final Functions functions;
    private final MemorySegment handle;
    private final Cleaner.Cleanable cleanable;
    /** add / result hold the read lock, close the write lock: a free never races a call. */
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private boolean closed;

    NativeRecommender(Functions functions, MemorySegment handle) {
        this.functions = functions;
        this.handle = handle;
        cleanable = CLEANER.register(this, new Free(functions.free(), handle));
    }

    /** The boolean pairs' true and false strings as two native arrays (NULL when empty). */
    static MemorySegment[] booleanPairs(Arena arena, List<BooleanPair> pairs) {
        return new MemorySegment[] {
            Native.cStrings(arena, pairs.stream().map(BooleanPair::trueValue).toList()),
            Native.cStrings(arena, pairs.stream().map(BooleanPair::falseValue).toList())};
    }

    /**
     * Adds {@code input}; it is consumed. See the subclass for what adding means.
     *
     * @throws IllegalArgumentException for invalid input (a duplicate or malformed column, a
     *                                  type change, a second frame for a one-shot recommender)
     * @throws AnalyticsException       for a failure inside the Rust kernels
     * @throws IllegalStateException    when closed
     */
    public void add(ArrowReader input, BufferAllocator allocator) {
        lock.readLock().lock();
        try {
            Native.run(() -> {
                try (ArrowArrayStream in = ArrowArrayStream.allocateNew(allocator);
                     Arena arena = Arena.ofConfined()) {
                    checkOpen();
                    Data.exportArrayStream(allocator, input, in);
                    MemorySegment error = arena.allocate(ADDRESS);
                    int code = (int) functions.add().invokeExact(
                        handle, MemorySegment.ofAddress(in.memoryAddress()), error);
                    if (code != 0) {
                        throw Native.failure(code, error.get(ADDRESS, 0));
                    }
                }
            });
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * One row per column (spec docs/superpowers/specs/2026-10-04-oneshot-recommender-design.md
     * §4); the state is kept. The returned reader holds one batch; the caller closes it.
     *
     * @throws AnalyticsException    for a failure inside the Rust kernels
     * @throws IllegalStateException when closed
     */
    public ArrowReader result(BufferAllocator allocator) {
        lock.readLock().lock();
        try {
            return Native.call(() -> {
                try (ArrowArrayStream out = ArrowArrayStream.allocateNew(allocator);
                     Arena arena = Arena.ofConfined()) {
                    checkOpen();
                    MemorySegment error = arena.allocate(ADDRESS);
                    int code = (int) functions.result().invokeExact(
                        handle, MemorySegment.ofAddress(out.memoryAddress()), error);
                    if (code != 0) {
                        throw Native.failure(code, error.get(ADDRESS, 0));
                    }
                    return Data.importArrayStream(allocator, out);
                }
            });
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Frees the native state, after any call in progress. Idempotent. */
    @Override
    public void close() {
        lock.writeLock().lock();
        try {
            if (!closed) {
                closed = true;
                cleanable.clean();
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException(getClass().getSimpleName() + " is closed");
        }
    }
}
