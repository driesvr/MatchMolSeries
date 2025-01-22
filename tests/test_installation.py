"""Test basic package installation and imports."""

import unittest

def test_import():
    """Test that the package can be imported."""
    try:
        from matchmolseries import MatchMolSeries
        assert True
    except ImportError:
        assert False, "Failed to import MatchMolSeries"

def test_version():
    """Test that version is available."""
    from matchmolseries import __version__
    assert __version__ is not None

if __name__ == "__main__":
    test_import()
    test_version()