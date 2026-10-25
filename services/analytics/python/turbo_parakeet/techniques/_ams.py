"""Parameter checks shared by the AMS entropy classes."""

# The sketch's error is roughly constant in bits (sd ≈ sqrt(Var / num_samples) ≈ 0.022 bits at
# 4 096 samples for a 1-bit column), so 0.1 bits ≈ 4.5σ; relative-only fails 1-bit columns ~1 seed in 7.
AMS_ATOL_BITS = 0.1

MAX_SAMPLES = 2**20  # keep in step with `ams_samples` in src/bindings/api/techniques.rs


def check_ams_params(num_samples: int, seed: int) -> None:
    if type(num_samples) is not int or not 1 <= num_samples <= MAX_SAMPLES:
        raise ValueError(
            f"num_samples must be an integer in [1, {MAX_SAMPLES}], got {num_samples!r}"
        )
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise ValueError(f"seed must be an integer in [0, 2**64 - 1], got {seed!r}")
