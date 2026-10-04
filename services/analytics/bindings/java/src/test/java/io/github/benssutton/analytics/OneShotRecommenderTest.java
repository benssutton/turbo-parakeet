package io.github.benssutton.analytics;

import static java.nio.charset.StandardCharsets.UTF_8;
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

class OneShotRecommenderTest {

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

    private static <T> T result(OneShotRecommender rec, BufferAllocator allocator,
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
    void recommendsToyData() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             OneShotRecommender rec = new OneShotRecommender(OneShotParams.defaults())) {
            rec.add(toyData(allocator), allocator);
            result(rec, allocator, root -> {
                assertEquals(List.of("a", "s"), column(root, "column"));
                assertEquals(List.of("computed", "computed"), column(root, "status"));
                assertEquals(List.of("uint8", "string"), column(root, "rec_arrow_type"));
                assertEquals(List.of("categorical", "boolean"), column(root, "class"));
                assertEquals(List.of("0", "x"), column(root, "min"));
                assertEquals(List.of("7", "y"), column(root, "max"));
                assertEquals(List.of("observed", "observed"), column(root, "est_method"));
                assertEquals(null, root.getVector("first_row"));
                return null;
            });
        }
    }

    @Test
    void resultIsRepeatable() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             OneShotRecommender rec = new OneShotRecommender(OneShotParams.defaults())) {
            rec.add(toyData(allocator), allocator);
            List<String> first = result(rec, allocator, root -> column(root, "rec_arrow_type"));
            assertEquals(first, result(rec, allocator, root -> column(root, "rec_arrow_type")));
        }
    }

    @Test
    void aSecondAddIsIllegalArgumentAndKeepsTheFirst() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             OneShotRecommender rec = new OneShotRecommender(OneShotParams.defaults())) {
            rec.add(toyData(allocator), allocator);
            ArrowReader again = toyData(allocator);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> rec.add(again, allocator));
            assertTrue(e.getMessage().contains("already been added"), e.getMessage());
            assertEquals(List.of("3", "3"), result(rec, allocator, root -> column(root, "n_rows")));
        }
    }

    @Test
    void duplicateColumnIsIllegalArgument() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             OneShotRecommender rec = new OneShotRecommender(OneShotParams.defaults())) {
            ArrowReader dup = reader(allocator, ints(allocator, "a", 1), strings(allocator, "a", "x"));
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> rec.add(dup, allocator));
            assertTrue(e.getMessage().contains("duplicate column \"a\""), e.getMessage());
        }
    }

    @Test
    void anEmptyRecommenderHasNoRows() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             OneShotRecommender rec = new OneShotRecommender(OneShotParams.defaults())) {
            assertEquals(0, (int) result(rec, allocator, VectorSchemaRoot::getRowCount));
        }
    }

    @Test
    void invalidParametersAreIllegalArguments() {
        OneShotParams d = OneShotParams.defaults();
        assertThrows(IllegalArgumentException.class, () -> new OneShotRecommender(
            new OneShotParams(d.categoricalThreshold(), 99, d.seed(), d.booleanPairs())));
        assertThrows(IllegalArgumentException.class, () -> new OneShotRecommender(
            new OneShotParams(d.categoricalThreshold(), d.zstdLevel(), d.seed(),
                List.of(new BooleanPair("Y", "y")))));
    }

    @Test
    void aClosedRecommenderRefusesWork() throws IOException {
        try (BufferAllocator allocator = new RootAllocator()) {
            OneShotRecommender rec = new OneShotRecommender(OneShotParams.defaults());
            rec.close();
            rec.close(); // idempotent
            try (ArrowReader input = toyData(allocator)) {
                assertThrows(IllegalStateException.class, () -> rec.add(input, allocator));
            }
            assertThrows(IllegalStateException.class, () -> rec.result(allocator));
        }
    }
}
