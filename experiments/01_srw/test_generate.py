"""Correctness checks for generate.py: shapes match the recipe, and
reproduce() (regenerate from recorded metadata) matches the original
persisted run exactly -- the same independent check generator.py's
`reproduce` provides for 00_synthetic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from generate import generate, reproduce


def test_shapes_match_recipe():
    samples = generate([16, 64, 256], [10, 20, 30], q=0.5, seed=0)
    assert samples[16].shape == (10,)
    assert samples[64].shape == (20,)
    assert samples[256].shape == (30,)


def test_reproduce_matches_saved_run(tmp_path):
    generate([16, 64, 256], 50, q=0.3, seed=123, out_dir=tmp_path, tag="run")
    original = np.load(tmp_path / "run.npz")

    reproduced = reproduce(tmp_path / "run.json")

    for k in [16, 64, 256]:
        assert np.array_equal(original[str(k)], reproduced[k])
