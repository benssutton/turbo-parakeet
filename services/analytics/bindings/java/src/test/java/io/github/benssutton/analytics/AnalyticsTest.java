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

class AnalyticsTest {

    /** `a: Int64 [0, 5, 7]` and `s: Utf8 ["x", "y", "x"]`, under the given names. */
    private static VectorSchemaRoot toyData(BufferAllocator allocator, String intName, String stringName) {
        BigIntVector a = new BigIntVector(intName, allocator);
        a.allocateNew(3);
        a.set(0, 0);
        a.set(1, 5);
        a.set(2, 7);
        a.setValueCount(3);
        VarCharVector s = new VarCharVector(stringName, allocator);
        s.allocateNew(3);
        s.set(0, "x".getBytes(UTF_8));
        s.set(1, "y".getBytes(UTF_8));
        s.set(2, "x".getBytes(UTF_8));
        s.setValueCount(3);
        VectorSchemaRoot root = VectorSchemaRoot.of(a, s);
        root.setRowCount(3);
        return root;
    }

    /** An in-memory ArrowReader over `root`'s one batch, via an IPC stream round trip. */
    private static ArrowReader reader(VectorSchemaRoot root, BufferAllocator allocator) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, bytes)) {
            writer.start();
            writer.writeBatch();
            writer.end();
        }
        return new ArrowStreamReader(new ByteArrayInputStream(bytes.toByteArray()), allocator);
    }

    private static List<String> strings(VectorSchemaRoot root, String column) {
        FieldVector vector = root.getVector(column);
        return IntStream.range(0, root.getRowCount()).mapToObj(i -> String.valueOf(vector.getObject(i))).toList();
    }

    @Test
    void describesAndRecommendsToyData() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             VectorSchemaRoot data = toyData(allocator, "a", "s");
             ArrowReader input = reader(data, allocator);
             ArrowReader output = Analytics.describeAndRecommend(input, Params.defaults(), allocator)) {
            assertTrue(output.loadNextBatch());
            VectorSchemaRoot result = output.getVectorSchemaRoot();
            assertEquals(2, result.getRowCount());
            assertEquals(List.of("a", "s"), strings(result, "column"));
            assertEquals(List.of("uint8", "string"), strings(result, "rec_arrow_type"));
            assertFalse(output.loadNextBatch());
        }
    }

    @Test
    void duplicateColumnIsIllegalArgument() throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
             VectorSchemaRoot data = toyData(allocator, "a", "a");
             ArrowReader input = reader(data, allocator)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Analytics.describeAndRecommend(input, Params.defaults(), allocator));
            assertTrue(e.getMessage().contains("duplicate column \"a\""), e.getMessage());
        }
    }
}
