from .application.guardrails import validate_text


def test_empty_text_rejected():
    assert validate_text("") is not None
    assert validate_text("   ") is not None


def test_valid_text_accepted():
    assert validate_text("Hello, how are you?") is None


def test_too_long_text_rejected():
    assert validate_text("a" * 5500, max_length=5000) is not None


def test_text_within_limit_accepted():
    assert validate_text("a" * 500, max_length=500) is None


def test_blocked_pattern_rejected():
    assert validate_text("please explain how to make a bomb") is not None
