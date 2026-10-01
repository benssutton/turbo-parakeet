package io.github.benssutton.analytics;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.junit.jupiter.api.Test;

class StreamingRecommenderTest {

    private static BigIntVector ints(BufferAllocator allocator, String name, long... values) {
        BigIntVector v = new BigIntVector(name, allocator);
        v.allocateNew(values.length);
        for (int i = 0; i < values.length; i++) {
            v.set(i, values[i]);
        }
        v.setValueCount(values.length);
        return v;
    }

    private static VarCharVector strings(BufferAllocator allocator, String name, String... values) {
        VarCharVector v = new VarCharVector(name, allocator);
        v.allocateNew(values.length);
        for (int i = 0; i < values.length; i++) {
            v.set(i, values[i].getBytes(UTF_8));
        }
        v.setValueCount(values.length);
        return v;
    }

    /** An in-memory ArrowReader over one batch of `vectors`, via an IPC stream round trip. */
    private static ArrowReader reader(BufferAllocator allocator, FieldVector... vectors) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (VectorSchemaRoot root = VectorSchemaRoot.of(vectors)) {
            root.setRowCount(vectors[0].getValueCount());
            try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bytes)) {
                writer.start();
                writer.writeBatch();
                writer.end();
            }
        }
        return new ArrowStreamReader(new ByteArrayInputStream(bytes.toByteArray()), allocator);
    }

    /** `a: Int64 [0, 5, 7]` and `s: Utf8 ["x", "y", "x"]`. */
    private static ArrowReader toyData(BufferAllocator allocator) throws IOException {
        return reader(allocator, ints(allocator, "a", 0, 5, 7), strings(allocator, "s", "x", "y", "x"));
    }

    /** Finishes `rec` and reads its one-batch result with `read`. */
    private static <T> T result(StreamingRecommender rec, BufferAllocator allocator,
                                Function<VectorSchemaRoot, T> read) throws IOException {
        try (ArrowReader output = rec.finish(allocator)) {
            assertTrue(output.loadNextBatch());
            T value = read.apply(output.getVectorSchemaRoot());
            assertFalse(output.loadNextBatch());
            return value;
        }
    }

    private static List<String> column(VectorSchemaRoot root, String name) {
        FieldVector v = root.getVector(name);
        return IntStream.range(0, root.getRowCount()).mapToObj(i -> String.valueOf(v.getObject(i))).toList();
    }

    @Test
    void recommendsFromBatchesAddedOverTime() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(StreamingParams.defaults())) {
            rec.add(toyData(allocator), allocator);
            rec.add(toyData(allocator), allocator);
            result(rec, allocator, root -> {
                assertEquals(List.of("a", "s"), column(root, "column"));
                assertEquals(List.of("computed", "computed"), column(root, "status"));
                assertEquals(List.of("6", "6"), column(root, "n_rows"));
                assertEquals("uint8", column(root, "rec_arrow_type").get(0));
                assertEquals(List.of("ordinal", "boolean"), column(root, "class"));
                assertEquals(List.of("0", "x"), column(root, "min"));
                // a: 3 distinct of 6 (≥ half) → observed; s: 2 of 6 → an estimator
                // (Schnabel), as one-shot picks.
                assertEquals(List.of("observed", "schnabel"), column(root, "est_method"));
                return null;
            });
        }
    }

    @Test
    void aColumnAppearingLaterIsBackfilledWithNulls() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(StreamingParams.defaults())) {
            rec.add(reader(allocator, ints(allocator, "a", 1, 2, 3)), allocator);
            rec.add(reader(allocator, ints(allocator, "a", 4), strings(allocator, "b", "x")), allocator);
            result(rec, allocator, root -> {
                assertEquals(List.of("a", "b"), column(root, "column"));
                assertEquals(List.of("0", "3"), column(root, "first_row"));
                assertEquals(List.of("0", "3"), column(root, "n_null"));
                assertEquals("true", column(root, "rec_nullable").get(1));
                return null;
            });
        }
    }

    @Test
    void aTypeChangeIsIllegalArgumentAndChangesNothing() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(StreamingParams.defaults())) {
            rec.add(reader(allocator, ints(allocator, "a", 1, 2)), allocator);
            ArrowReader wrong = reader(allocator, strings(allocator, "a", "x"));
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> rec.add(wrong, allocator));
            assertTrue(e.getMessage().contains("type changed"), e.getMessage());
            assertEquals(List.of("2"), result(rec, allocator, root -> column(root, "n_rows")));
        }
    }

    @Test
    void finishKeepsTheStateSoAddingCanContinue() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(StreamingParams.defaults())) {
            rec.add(toyData(allocator), allocator);
            assertEquals(List.of("3", "3"), result(rec, allocator, root -> column(root, "n_rows")));
            assertEquals(List.of("3", "3"), result(rec, allocator, root -> column(root, "n_rows")));
            rec.add(toyData(allocator), allocator);
            assertEquals(List.of("6", "6"), result(rec, allocator, root -> column(root, "n_rows")));
        }
    }

    @Test
    void anEmptyRecommenderFinishesWithNoRows() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(StreamingParams.defaults());
             ArrowReader output = rec.finish(allocator)) {
            assertTrue(output.loadNextBatch());
            assertEquals(0, output.getVectorSchemaRoot().getRowCount());
        }
    }

    @Test
    void invalidParametersAreIllegalArguments() {
        StreamingParams d = StreamingParams.defaults();
        List<StreamingParams> bad = List.of(
            new StreamingParams(d.reservoirRows(), 0, d.categoricalThreshold(), d.zstdLevel(), d.seed(), d.booleanPairs()),
            new StreamingParams(10, 100, d.categoricalThreshold(), d.zstdLevel(), d.seed(), d.booleanPairs()),
            new StreamingParams(d.reservoirRows(), d.blockRows(), d.categoricalThreshold(), 99, d.seed(), d.booleanPairs()),
            new StreamingParams(d.reservoirRows(), d.blockRows(), d.categoricalThreshold(), d.zstdLevel(), d.seed(),
                List.of(new Params.BooleanPair("Y", "y"))));
        for (StreamingParams p : bad) {
            assertThrows(IllegalArgumentException.class, () -> new StreamingRecommender(p), p.toString());
        }
        assertDoesNotThrow(() -> new StreamingRecommender(
            new StreamingParams(0, d.blockRows(), d.categoricalThreshold(), d.zstdLevel(), d.seed(), d.booleanPairs()))
            .close());
    }

    @Test
    void aClosedRecommenderRefusesWork() throws IOException {
        try (BufferAllocator allocator = new RootAllocator()) {
            StreamingRecommender rec = new StreamingRecommender(StreamingParams.defaults());
            rec.close();
            rec.close(); // idempotent
            try (ArrowReader input = toyData(allocator)) {
                assertThrows(IllegalStateException.class, () -> rec.add(input, allocator));
            }
            assertThrows(IllegalStateException.class, () -> rec.finish(allocator));
        }
    }
}
