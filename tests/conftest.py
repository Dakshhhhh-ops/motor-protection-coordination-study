"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models import Plant  # noqa: E402

PLANT_YAML = ROOT / "data" / "plant.yaml"
CABLES_YAML = ROOT / "data" / "cables.yaml"


@pytest.fixture(scope="session")
def raw_plant() -> dict:
    return yaml.safe_load(PLANT_YAML.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def plant() -> Plant:
    """The base-case plant, loaded once for the whole session."""
    return Plant.from_yaml(PLANT_YAML, CABLES_YAML)


@pytest.fixture
def make_plant(tmp_path, raw_plant):
    """Build a Plant from the base data with arbitrary edits applied.

    Used to prove that each acceptance check actually fires on bad input: a
    check that cannot fail is worthless, so every one of them is exercised
    against a deliberately broken plant.
    """

    def _make(mutate=None, scenario=None) -> Plant:
        data = copy.deepcopy(raw_plant)
        if mutate is not None:
            mutate(data)
        path = tmp_path / "plant.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return Plant.from_yaml(path, CABLES_YAML, scenario=scenario)

    return _make


def motor_dict(data: dict, tag: str) -> dict:
    """Locate one motor's raw mapping inside the plant data."""
    for m in data["motors"]:
        if m["tag"] == tag:
            return m
    raise KeyError(tag)
