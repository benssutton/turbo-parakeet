package io.github.benssutton.analytics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The error conventions and plumbing shared by the bindings (the C ABI itself is covered by their tests). */
class NativeTest {

    @Test
    void libraryDirectoryMustBeSet() {
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> Native.libraryPath(null));
        assertTrue(e.getMessage().contains("analytics.library.dir"), e.getMessage());
    }

    @Test
    void missingLibraryNamesTheExpectedPath(@TempDir Path empty) {
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> Native.libraryPath(empty.toString()));
        assertTrue(e.getMessage().contains(System.mapLibraryName("analytics")), e.getMessage());
    }

    @Test
    void libraryIsFoundInTheConfiguredDirectory() {
        Path library = Native.libraryPath(System.getProperty("analytics.library.dir"));
        assertTrue(Files.isRegularFile(library), library.toString());
    }

    @Test
    void codeOneIsInvalidInputAndOtherCodesAreRuntimeFailures() throws Throwable {
        // A null message (nothing to free) falls back to the code.
        RuntimeException invalid = Native.failure(1, MemorySegment.NULL);
        assertInstanceOf(IllegalArgumentException.class, invalid);
        assertEquals("analytics error code 1", invalid.getMessage());

        RuntimeException compute = Native.failure(2, MemorySegment.NULL);
        assertEquals(AnalyticsException.class, compute.getClass());
        assertEquals("analytics error code 2", compute.getMessage());
    }

    @Test
    void callWrapsCheckedFailuresAndPassesRuntimeOnesThrough() {
        Exception checked = new IOException("boom");
        RuntimeException wrapped = assertThrows(AnalyticsException.class, () -> Native.call(() -> {
            throw checked;
        }));
        assertSame(checked, wrapped.getCause());

        IllegalArgumentException invalid = new IllegalArgumentException("bad");
        assertSame(invalid, assertThrows(IllegalArgumentException.class, () -> Native.run(() -> {
            throw invalid;
        })));

        AssertionError error = new AssertionError("fatal");
        assertSame(error, assertThrows(AssertionError.class, () -> Native.call(() -> {
            throw error;
        })));
    }

    @Test
    void callReturnsTheResult() {
        assertEquals(42, Native.call(() -> 42));
    }

    @Test
    void emptyStringListIsNullAndOthersAreAnArray() {
        try (Arena arena = Arena.ofConfined()) {
            assertEquals(MemorySegment.NULL, Native.cStrings(arena, List.of()));
            assertNotEquals(MemorySegment.NULL, Native.cStrings(arena, List.of("a", "b")));
        }
    }
}
