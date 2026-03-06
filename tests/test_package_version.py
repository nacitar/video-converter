from video_converter import __version__


def test_version_defined() -> None:
    assert bool(__version__)
