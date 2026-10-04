package io.github.benssutton.analytics;

/**
 * A failure inside the native analytics library (a compute error in the Rust kernels, return
 * code 2 of the C ABI) or while calling it. Invalid input is an {@link IllegalArgumentException}
 * instead.
 */
public final class AnalyticsException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    AnalyticsException(String message) {
        super(message);
    }

    AnalyticsException(Throwable cause) {
        super(cause);
    }
}
