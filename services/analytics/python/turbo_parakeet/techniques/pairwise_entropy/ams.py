import math

import polars as pl

from turbo_parakeet import _plugin
from turbo_parakeet.techniques._ams import check_ams_params
from turbo_parakeet.techniques.base import group_by_frame, metric_mismatches
from turbo_parakeet.techniques.pairwise_entropy.base import PairwiseEntropy


class PairwiseEntropyAmsRust(PairwiseEntropy):
    """Rust extension functions `ams_marginal_entropy` + `ams_pairwise_joint_entropy`: one
    Chakrabarti–Cormode–McGregor sketch (`num_samples` estimators, O(num_samples) memory)
    per column and per pair, rayon-parallel. Entropies agree with the reference within
    RTOL; MI / NMI within the error the entropies propagate (see `agreement`)."""

    # mixed_dtypes(1_000) at num_samples=4096, seed=0: max relative error 2.7% (h_ab).
    RTOL = 0.05
    ATOL = 1e-9

    def __init__(self, *, num_samples: int = 4096, seed: int = 0, **thresholds):
        super().__init__(**thresholds)
        check_ams_params(num_samples, seed)
        self.num_samples = num_samples
        self.seed = seed

    def _compute(self, frames, combos):
        parts = []
        for frame, group in group_by_frame(combos).items():
            df = frames[frame].select(
                list(dict.fromkeys(c for k in group for _, c in k))
            )
            marginal = dict(
                _plugin.ams_marginal_entropy(
                    df, self.num_samples, self.seed
                ).iter_rows()
            )
            joint = _plugin.ams_pairwise_joint_entropy(
                df,
                [(a, b) for (_, a), (_, b) in group],
                self.num_samples,
                self.seed,
            )
            h_ab = {frozenset((a, b)): h for a, b, h in joint.iter_rows()}
            parts.append(
                self.entropy_rows(
                    group,
                    [marginal[a] for (_, a), _ in group],
                    [marginal[b] for _, (_, b) in group],
                    [h_ab[frozenset((a, b))] for (_, a), (_, b) in group],
                    [df.height] * len(group),
                )
            )
        return pl.concat(parts)

    def agreement(self, result: pl.DataFrame, reference: pl.DataFrame) -> list[str]:
        """Entropies within RTOL. MI within RTOL·(h_a + h_b + h_ab) of the reference and NMI
        within that bound ÷ min(h_a, h_b): MI is a difference of estimates, so its relative
        error is unbounded. Conclusions are not compared: an estimate within tolerance can
        still land on the other side of a threshold."""
        keys = self.key_columns()
        problems = metric_mismatches(
            result,
            reference,
            keys,
            ["h_a", "h_b", "h_ab"],
            self.RTOL,
            self.ATOL,
        )
        if problems[:1] == ["key columns differ from the reference"]:
            return problems
        problems += metric_mismatches(result, reference, keys, ["n_rows"], 0.0, 0.0)
        both = (
            reference.select(*keys, "h_a", "h_b", "h_ab", "mi", "nmi")
            .with_columns(got_mi=result["mi"], got_nmi=result["nmi"])
            .filter(pl.col("h_ab").is_not_null())
        )
        for r in both.iter_rows(named=True):
            label = " ~ ".join(
                f"{r[keys[i]]}.{r[keys[i + 1]]}" for i in range(0, len(keys), 2)
            )
            bound = self.RTOL * (r["h_a"] + r["h_b"] + r["h_ab"]) + self.ATOL
            if r["got_mi"] is None or not abs(r["got_mi"] - r["mi"]) <= bound:
                problems.append(
                    f"{label}: mi {r['got_mi']!r} != {r['mi']!r} ± {bound:.3g}"
                )
            if math.isnan(r["nmi"]):
                continue
            nmi_bound = bound / min(r["h_a"], r["h_b"])
            if r["got_nmi"] is None or not abs(r["got_nmi"] - r["nmi"]) <= nmi_bound:
                problems.append(
                    f"{label}: nmi {r['got_nmi']!r} != {r['nmi']!r} ± {nmi_bound:.3g}"
                )
        return problems
