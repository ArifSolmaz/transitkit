"""Basic tests for TransitKit"""

from transitkit import __version__, hello

def test_version():
    """Test version is set"""
    assert __version__ == "0.1.0"

def test_hello():
    """Test hello function"""
    result = hello()
    assert "TransitKit" in result
    assert "0.1.0" in result