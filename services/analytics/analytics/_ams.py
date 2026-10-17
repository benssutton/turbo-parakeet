"""Parameter checks shared by the AMS entropy classes."""

MAX_SAMPLES = 2**20  # keep in step with `ams_samples` in src/bindings/api/techniques.rs


def check_ams_params(num_samples: int, seed: int) -> None:
    if type(num_samples) is not int or not 1 <= num_samples <= MAX_SAMPLES:
        raise ValueError(
            f"num_samples must be an integer in [1, {MAX_SAMPLES}], got {num_samples!r}"
        )
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise ValueError(f"seed must be an integer in [0, 2**64 - 1], got {seed!r}")
