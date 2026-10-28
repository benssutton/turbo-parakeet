import polars as pl
import pytest

from evaluate.registry import ENTRIES, GROUPS, RegistryError, resolve


def test_names_resolve_with_default_rows():
    (d,) = resolve(["gcd"])
    assert (d.name, d.rows) == ("gcd", 1_000_000)


def test_rows_override_on_a_scalable_entry():
    (d,) = resolve(["synth_frame:2000"])
    assert (d.name, d.rows) == ("synth_frame", 2000)
    assert d.load().height == 2000


def test_rows_override_rejected_on_a_fixed_entry():
    with pytest.raises(RegistryError, match="synth_small has a fixed size"):
        resolve(["synth_small:100"])


def test_groups_expand_in_order():
    assert [d.name for d in resolve(["calibration"])] == [
        "synth_frame",
        "gcd",
        "delta",
        "skip",
    ]
    assert GROUPS["calibration"] == ("synth_frame", "gcd", "delta", "skip")


def test_all_lists_every_entry_once():
    assert [d.name for d in resolve(["--all"])] == list(ENTRIES)


def test_unknown_name_lists_the_registry():
    with pytest.raises(RegistryError, match="unknown dataset 'nope'.*gcd"):
        resolve(["nope"])


def test_ipc_path_is_a_dataset(tmp_path):
    p = tmp_path / "My Frame.arrow"
    pl.DataFrame({"a": [1, 2, 3]}).write_ipc(p)
    (d,) = resolve([str(p)])
    assert d.name == "my_frame" and d.rows == 3
    assert d.load()["a"].to_list() == [1, 2, 3]


def test_unreadable_ipc_path(tmp_path):
    p = tmp_path / "bad.arrow"
    p.write_bytes(b"not arrow")
    with pytest.raises(RegistryError, match="cannot read"):
        resolve([str(p)])


def test_duplicates_are_rejected():
    with pytest.raises(RegistryError, match="twice"):
        resolve(["gcd", "calibration"])
