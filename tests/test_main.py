from coreason_signal.main import hello_world


def test_hello_world() -> None:
    """Test the hello_world function."""
    assert hello_world() == "Hello World!"
