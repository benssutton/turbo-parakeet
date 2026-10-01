package io.github.benssutton.analytics;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

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
 * Recommend's dtype recommendations from batches added over time; all state stays in Rust
 * (services/analytics/src/streaming.rs, through the C ABI's {@code analytics_recommender_*}).
 * Spec: docs/superpowers/specs/2026-09-29-streaming-recommender-design.md.
 *
 * <p>{@link #add} any number of times — columns may appear, disappear (their rows count as
 * null) or start as the Null type; any other type change is an {@link IllegalArgumentException}.
 * {@link #finish} at any point returns one row per column and keeps the state.
 *
 * <p>Thread safety: {@code add} and {@code finish} may be called from several threads (the
 * native side serialises them); {@link #close} waits for calls in progress. A producer whose
 * callbacks re-enter this recommender while {@code add} pulls its stream deadlocks.
 */
public final class StreamingRecommender implements AutoCloseable {

    private static final MethodHandle NEW = Native.handle("analytics_recommender_new",
        FunctionDescriptor.of(JAVA_INT,
            JAVA_LONG,  // uint64_t reservoir_rows
            JAVA_LONG,  // uint64_t block_rows
            JAVA_LONG,  // uint64_t categorical_threshold
            JAVA_INT,   // int32_t zstd_level
            JAVA_LONG,  // uint64_t seed
            ADDRESS,    // const char *const *bool_true
            ADDRESS,    // const char *const *bool_false
            JAVA_LONG,  // size_t n_bool_pairs
            ADDRESS,    // AnalyticsRecommender **out
            ADDRESS));  // char **error
    private static final MethodHandle ADD = Native.handle("analytics_recommender_add",
        FunctionDescriptor.of(JAVA_INT,
            ADDRESS,    // AnalyticsRecommender *h
            ADDRESS,    // ArrowArrayStream *batches (consumed)
            ADDRESS));  // char **error
    private static final MethodHandle FINISH = Native.handle("analytics_recommender_finish",
        FunctionDescriptor.of(JAVA_INT,
            ADDRESS,    // AnalyticsRecommender *h
            ADDRESS,    // ArrowArrayStream *out
            ADDRESS));  // char **error
    private static final MethodHandle FREE = Native.handle("analytics_recommender_free",
        FunctionDescriptor.ofVoid(ADDRESS));

    private static final Cleaner CLEANER = Cleaner.create();

    /** Frees the native recommender; run once, by {@link #close} or when unreachable. */
    private record Free(MemorySegment handle) implements Runnable {
        @Override
        public void run() {
            try {
                FREE.invokeExact(handle);
            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }
    }

    private final MemorySegment handle;
    private final Cleaner.Cleanable cleanable;
    /** add / finish hold the read lock, close the write lock: a free never races a call. */
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private boolean closed;

    /** @throws IllegalArgumentException for invalid parameters (e.g. {@code blockRows} 0) */
    public StreamingRecommender(StreamingParams params) {
        try (Arena arena = Arena.ofConfined()) {
            List<Params.BooleanPair> pairs = params.booleanPairs();
            MemorySegment trues = Native.cStrings(arena, pairs.stream().map(Params.BooleanPair::trueValue).toList());
            MemorySegment falses = Native.cStrings(arena, pairs.stream().map(Params.BooleanPair::falseValue).toList());
            MemorySegment out = arena.allocate(ADDRESS);
            MemorySegment error = arena.allocate(ADDRESS);  // zero-initialised: null
            int code = (int) NEW.invokeExact(
                params.reservoirRows(),
                params.blockRows(),
                params.categoricalThreshold(),
                params.zstdLevel(),
                params.seed(),
                trues,
                falses,
                (long) pairs.size(),
                out,
                error);
            if (code != 0) {
                throw Native.failure(code, error.get(ADDRESS, 0));
            }
            handle = out.get(ADDRESS, 0);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
        cleanable = CLEANER.register(this, new Free(handle));
    }

    /**
     * Adds every batch of {@code input}, in order; {@code input} is consumed. Each batch is
     * atomic (on failure the state is as before that batch), the call is not: batches before
     * a failing one stay added.
     *
     * @throws IllegalArgumentException for invalid input (a type change, a duplicate or
     *                                  malformed column)
     * @throws RuntimeException         for a failure inside the Rust kernels
     * @throws IllegalStateException    when closed
     */
    public void add(ArrowReader input, BufferAllocator allocator) {
        lock.readLock().lock();
        try (ArrowArrayStream in = ArrowArrayStream.allocateNew(allocator);
             Arena arena = Arena.ofConfined()) {
            checkOpen();
            Data.exportArrayStream(allocator, input, in);
            MemorySegment error = arena.allocate(ADDRESS);
            int code = (int) ADD.invokeExact(handle, MemorySegment.ofAddress(in.memoryAddress()), error);
            if (code != 0) {
                throw Native.failure(code, error.get(ADDRESS, 0));
            }
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * One row per column seen so far, with the columns of the parity spec
     * (docs/superpowers/specs/2026-10-01-recommender-parity-design.md §7: {@code column, status,
     * dtype, first_row, n_rows, n_null}, Describe's value block, {@code n_midnight}, the sizes,
     * {@code rec_*}, {@code n_sampled_rows, n_sampled_blocks}); the state is kept, so adding can
     * continue. The returned reader holds one batch; the caller closes it.
     *
     * @throws RuntimeException      for a failure inside the Rust kernels
     * @throws IllegalStateException when closed
     */
    public ArrowReader finish(BufferAllocator allocator) {
        lock.readLock().lock();
        try (ArrowArrayStream out = ArrowArrayStream.allocateNew(allocator);
             Arena arena = Arena.ofConfined()) {
            checkOpen();
            MemorySegment error = arena.allocate(ADDRESS);
            int code = (int) FINISH.invokeExact(handle, MemorySegment.ofAddress(out.memoryAddress()), error);
            if (code != 0) {
                throw Native.failure(code, error.get(ADDRESS, 0));
            }
            return Data.importArrayStream(allocator, out);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
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
            throw new IllegalStateException("StreamingRecommender is closed");
        }
    }
}
