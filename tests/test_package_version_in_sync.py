import tomllib
from pathlib import Path

from video_converter import __version__


def find_pyproject_toml() -> Path:
    """Walks up from this test's directory to find pyproject.toml."""
    for parent in Path(__file__).parents:
        path = parent / "pyproject.toml"
        if path.is_file():
            return path
    msg = "pyproject.toml not found"
    raise FileNotFoundError(msg)


def test_version_matches_pyproject() -> None:
    """
    Test that the hardcoded version in __init__.py matches the version
    in pyproject.toml.

    This ensures that the code's version and the declared package version
    are always in sync.
    """
    with find_pyproject_toml().open("rb") as handle:
        assert __version__ == tomllib.load(handle)["project"]["version"]
