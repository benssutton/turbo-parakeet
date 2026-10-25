"""The Arrow IPC cache for generated datasets.

The cache is Arrow IPC in the file format (Feather v2) uncompressed, at
`<cache_dir>/<dataset_id>.arrow`, with the manifest as JSON beside it. IPC
keeps the exact Arrow type and buffers, so the cached column is bit-identical
to what the loader inserts, with no encode or decode step.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from synthdata.column import metrics, realdata
from synthdata.column.spec import DatasetSpec, SpecError
from synthdata.generate import COLUMN_NAME, Dataset, Manifest, generate, observed

__all__ = ["save", "load_path", "load_or_generate"]


def _paths(dataset_id: str, cache_dir: Path) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{dataset_id}.arrow", cache_dir / f"{dataset_id}.json"


def save(dataset: Dataset, cache_dir: Path) -> Path:
    """Write the array as an uncompressed Arrow IPC file (Feather v2) and the
    manifest as JSON beside it. Returns the `.arrow` path."""
    array_path, manifest_path = _paths(dataset.manifest.dataset_id, cache_dir)
    array_path.parent.mkdir(parents=True, exist_ok=True)

    batch = pa.record_batch([dataset.array], names=[COLUMN_NAME])
    options = pa.ipc.IpcWriteOptions(compression=None)
    with pa.OSFile(str(array_path), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema, options=options) as writer:
            writer.write_batch(batch)

    manifest_path.write_text(dataset.manifest.to_json(), encoding="utf-8")
    return array_path


def _load(array_path: Path, manifest_path: Path) -> Dataset:
    # read_all() copies out of the file so the handle can close immediately;
    # a memory map would pin the file open, which matters on Windows.
    with pa.OSFile(str(array_path), "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    column = table.column(COLUMN_NAME)
    array = column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
    manifest = Manifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    return Dataset(array=array, manifest=manifest)


def load_path(spec: DatasetSpec) -> Dataset:
    """A real-data dataset (`dataset: {path, column}`): the column of
    `spec.path`, unchanged, with its metrics measured by `chbench.datagen.metrics`
    on the non-null values (value ids, so every Arrow type is measured alike).
    Nothing was targeted: `targets`, `within_tolerance`, `solves` and `warnings`
    are empty. SpecError if the file no longer matches `spec.content_hash`."""
    spec.validate()
    started = time.perf_counter()
    if realdata.file_digest(spec.path, spec.column) != spec.content_hash:
        raise SpecError(
            f"dataset file {spec.path!r} has changed since its spec "
            "was measured (content_hash differs); rebuild the spec"
        )
    array = realdata.read_column(spec.path, spec.column)
    ids = pc.dictionary_encode(
        realdata.widen_decimal(array.drop_null())
    ).indices.to_numpy()
    achieved = {
        "cardinality_ratio": float(metrics.cardinality_ratio(ids)),
        "entropy_bits": float(metrics.entropy_bits(ids)),
    }
    manifest = Manifest(
        dataset_id=spec.dataset_id,
        spec=spec,
        achieved=achieved,
        targets={},
        within_tolerance={},
        solves={},
        warnings=[],
        **observed(array),
        generated_at=datetime.now(timezone.utc).isoformat(),
        generation_seconds=time.perf_counter() - started,
    )
    return Dataset(array=array, manifest=manifest)


def load_or_generate(spec: DatasetSpec, cache_dir: Path) -> Dataset:
    """Return the cached dataset for `spec.dataset_id` if both files exist,
    otherwise generate it and save it first. A path spec is read from its file
    and never cached (`load_path`)."""
    if spec.path is not None:
        return load_path(spec)
    array_path, manifest_path = _paths(spec.dataset_id, cache_dir)
    if array_path.exists() and manifest_path.exists():
        return _load(array_path, manifest_path)
    dataset = generate(spec)
    save(dataset, cache_dir)
    return dataset
