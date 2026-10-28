"""
Membership (containment) speed benchmark: BloomRust vs BloomFastbloom vs MembershipExact (reference).

Run: python tests/performance/benchmark_membership.py
"""

from harness import Dataset, synth_dataset, run

from synthdata.scenarios import related_frames, similar_frames

if __name__ == "__main__":
    run(
        "turbo_parakeet.techniques.membership",
        [
            synth_dataset(),
            Dataset(
                "narrow related 3 frames x 1M rows", lambda: related_frames(1_000_000)
            ),
            Dataset(
                "narrow related 3 frames x 50k rows", lambda: related_frames(50_000)
            ),
            Dataset(
                "wide 2 frames x 100 cols",
                lambda: similar_frames(
                    n_similar=50, n_independent=50, col_size=5_000, n_elements=100_000
                ),
            ),
        ],
    )
