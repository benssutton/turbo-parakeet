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
import java.util.Map;
import java.util.OptionalLong;
import java.util.function.Function;
import java.util.stream.IntStream;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.complex.StructVector;
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

    /** Reads the one-batch result of `rec` with `read`. */
    private static <T> T result(StreamingRecommender rec, BufferAllocator allocator,
                                Function<VectorSchemaRoot, T> read) throws IOException {
        try (ArrowReader output = rec.result(allocator)) {
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
                assertEquals("uint8", column(root, "datafusion_type").get(0));
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
                assertEquals("true", column(root, "nullable").get(1));
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
    void resultKeepsTheStateSoAddingCanContinue() throws IOException {
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
    void anEmptyRecommenderHasNoRows() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(StreamingParams.defaults());
             ArrowReader output = rec.result(allocator)) {
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
                List.of(new BooleanPair("Y", "y"))));
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
            assertThrows(IllegalStateException.class, () -> rec.result(allocator));
        }
    }

    @Test
    void topKCountsEveryBatch() throws IOException {
        assertEquals(OptionalLong.of(256), StreamingParams.defaults().topK());
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(StreamingParams.defaults())) {
            rec.add(toyData(allocator), allocator);
            rec.add(toyData(allocator), allocator);
            result(rec, allocator, root -> {
                MapVector v = (MapVector) root.getVector("top_k");
                assertTrue(v.isNull(0));
                Map<?, ?> first = (Map<?, ?>) v.getObject(1).get(0);
                assertEquals("x", first.get("key").toString());
                assertEquals(4L, ((Number) first.get("value")).longValue());
                return null;
            });
        }
    }

    @Test
    void topKLimitReachesTheNativeConstructor() throws IOException {
        StreamingParams d = StreamingParams.defaults();
        StreamingParams params = new StreamingParams(d.reservoirRows(), d.blockRows(), d.categoricalThreshold(),
            d.zstdLevel(), d.seed(), d.booleanPairs(), OptionalLong.of(1));
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(params)) {
            rec.add(toyData(allocator), allocator);
            rec.add(toyData(allocator), allocator);
            result(rec, allocator, root -> {
                MapVector v = (MapVector) root.getVector("top_k");
                assertEquals(1, v.getObject(1).size());
                Map<?, ?> only = (Map<?, ?>) v.getObject(1).get(0);
                assertEquals("x", only.get("key").toString());
                assertEquals(4L, ((Number) only.get("value")).longValue());
                return null;
            });
        }
    }

    /** `k: Int64 [10, 20, 30, 40]`, `a: Int64 [1, 1, 2, 2]`. */
    private static ArrowReader idData(BufferAllocator allocator) throws IOException {
        BigIntVector a = new BigIntVector("a", allocator);
        a.allocateNew(4);
        a.set(0, 1);
        a.set(1, 1);
        a.set(2, 2);
        a.set(3, 2);
        a.setValueCount(4);
        return reader(allocator, ints(allocator, "k", 10, 20, 30, 40), a);
    }

    private static StreamingParams withIds(List<String> ids) {
        StreamingParams d = StreamingParams.defaults();
        return new StreamingParams(d.reservoirRows(), d.blockRows(), d.categoricalThreshold(), d.zstdLevel(),
            d.seed(), d.booleanPairs(), d.topK(), ids, EntropyEstimator.AMS);
    }

    @Test
    void amsEstimatorAndIdColumnsOnASmallFrame() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(withIds(List.of("k")))) {
            rec.add(idData(allocator), allocator);
            result(rec, allocator, root -> {
                assertEquals(List.of("exact", "exact"), column(root, "entropy_method"));
                assertEquals(2.0, ((Float8Vector) root.getVector("entropy")).get(0), 1e-12);  // k: 4 distinct
                assertEquals(1.0, ((Float8Vector) root.getVector("entropy")).get(1), 1e-12);  // a: two halves
                assertEquals(List.of("0", "0"), column(root, "exemplar_row"));
                StructVector id = (StructVector) root.getVector("exemplar_id");
                assertEquals(10L, id.getChild("k", BigIntVector.class).get(1));
                assertEquals(List.of("null", "null"), column(root, "null_exemplar_row"));
                return null;
            });
        }
    }

    @Test
    void aMissingIdColumnIsIllegalArgument() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(withIds(List.of("nope")))) {
            ArrowReader data = idData(allocator);
            assertThrows(IllegalArgumentException.class, () -> rec.add(data, allocator));
        }
    }

    @Test
    void compressionKeywordsAreAccepted() throws IOException {
        StreamingParams d = StreamingParams.defaults();
        StreamingParams p = new StreamingParams(d.reservoirRows(), d.blockRows(), d.categoricalThreshold(),
            d.zstdLevel(), d.seed(), d.booleanPairs(), d.topK(), d.idColumns(), d.entropyEstimator(),
            List.of(3), 0.02, 0.2, 0.02);
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(p)) {
            rec.add(reader(allocator, ints(allocator, "a", 0, 5, 7)), allocator);
            result(rec, allocator, root -> {
                assertEquals(1, root.getRowCount());
                return null;
            });
        }
        StreamingParams bad = new StreamingParams(d.reservoirRows(), d.blockRows(), d.categoricalThreshold(),
            d.zstdLevel(), d.seed(), d.booleanPairs(), d.topK(), d.idColumns(), d.entropyEstimator(),
            List.of(3), 0.01, 1.0, 0.01);
        assertThrows(IllegalArgumentException.class, () -> new StreamingRecommender(bad));
    }

    @Test
    void skipIndexKeywordsAreAccepted() throws IOException {
        StreamingParams d = StreamingParams.defaults();
        StreamingParams p = new StreamingParams(d.reservoirRows(), d.blockRows(), d.categoricalThreshold(),
            d.zstdLevel(), d.seed(), d.booleanPairs(), d.topK(), d.idColumns(), d.entropyEstimator(),
            d.zstdLevels(), 0.01, 0.10, 0.01, List.of(1, 2), List.of(125L), List.of(0.025), 0.4, 0.2, 0.2);
        try (BufferAllocator allocator = new RootAllocator();
             StreamingRecommender rec = new StreamingRecommender(p)) {
            rec.add(reader(allocator, ints(allocator, "a", 0, 5, 7)), allocator);
            result(rec, allocator, root -> {
                assertEquals(1, root.getRowCount());
                return null;
            });
        }
        assertThrows(IllegalArgumentException.class, () -> new StreamingRecommender(new StreamingParams(
            d.reservoirRows(), d.blockRows(), d.categoricalThreshold(), d.zstdLevel(), d.seed(), d.booleanPairs(),
            d.topK(), d.idColumns(), d.entropyEstimator(), d.zstdLevels(), 0.01, 0.10, 0.01,
            List.of(1), List.of(125L), List.of(0.025), 1.5, 0.1, 0.1)));
    }
}
