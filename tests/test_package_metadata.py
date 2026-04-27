from __future__ import annotations

import importlib.metadata
import tomllib
from functools import cache
from pathlib import Path


@cache
def find_pyproject_toml() -> Path:
    """Walks up from this test's directory to find pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        path = parent / "pyproject.toml"
        if path.is_file():
            return path
    msg = "pyproject.toml not found"
    raise FileNotFoundError(msg)


@cache
def load_project_name() -> str:
    with find_pyproject_toml().open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project")
    if not isinstance(project, dict):
        msg = "project table not found in pyproject.toml"
        raise KeyError(msg)
    name = project.get("name")
    if not isinstance(name, str) or not name:
        msg = "project.name not found in pyproject.toml"
        raise KeyError(msg)
    return name


def test_distribution_version_is_available() -> None:
    version = importlib.metadata.version(load_project_name())
    assert version
