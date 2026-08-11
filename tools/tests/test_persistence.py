"""Round-trip and determinism checks for tools/persistence.py, the
sample+metadata persistence shared by experiments/00_synthetic/generator.py
and experiments/01_srw/generate.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from persistence import (
    content_id,
    load_metadata,
    load_samples,
    normalize_scales_n,
    save_samples,
    write_metadata,
)


def test_normalize_scales_n_scalar_broadcasts():
    scales, n = normalize_scales_n([4, 8, 16], 100)
    assert scales == [4, 8, 16]
    assert n == [100, 100, 100]


def test_normalize_scales_n_sequence_matches():
    scales, n = normalize_scales_n([4, 8, 16], [10, 20, 30])
    assert n == [10, 20, 30]


def test_normalize_scales_n_length_mismatch_raises():
    with pytest.raises(ValueError):
        normalize_scales_n([4, 8, 16], [10, 20])


def test_save_load_samples_roundtrip(tmp_path):
    samples = {4: np.array([1.0, 2.0, 3.0]), 8: np.array([4.0, 5.0])}
    path = save_samples(tmp_path / "run.npz", samples)
    loaded = load_samples(path)
    assert set(loaded) == {4, 8}
    assert np.array_equal(loaded[4], samples[4])
    assert np.array_equal(loaded[8], samples[8])


def test_load_samples_missing_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_samples(tmp_path / "nope.npz")


def test_write_load_metadata_roundtrip(tmp_path):
    path = write_metadata(
        path=tmp_path / "run.json",
        params={"q": 0.5, "model": "srw"},
        scales=[4, 8],
        n=[10, 10],
        seed=123,
        timing_seconds={4: 0.001, 8: 0.002},
    )
    meta = load_metadata(path)
    assert meta["params"] == {"q": 0.5, "model": "srw"}
    assert meta["seed"] == 123
    assert meta["timing_seconds"] == {"4": 0.001, "8": 0.002}


def test_load_metadata_missing_returns_none(tmp_path):
    assert load_metadata(tmp_path / "nope.npz") is None


def test_content_id_deterministic_and_sensitive_to_content():
    a = content_id({"q": 0.5}, [4, 8], [10, 10], 1)
    b = content_id({"q": 0.5}, [4, 8], [10, 10], 1)
    c = content_id({"q": 0.6}, [4, 8], [10, 10], 1)
    assert a == b
    assert a != c
