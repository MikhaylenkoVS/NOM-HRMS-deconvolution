"""Unit tests for _safety.py: safe() and _safe_df() — Optional-type wrappers."""

import pandas as pd
from src.core._safety import safe, _safe_df


class TestSafe:
    """safe(obj, default) — return obj if not None, else default."""

    def test_returns_obj_when_not_none(self):
        assert safe(42, 0) == 42
        assert safe("hello", "fallback") == "hello"
        assert safe([1, 2], []) == [1, 2]

    def test_returns_default_when_none(self):
        assert safe(None, 0) == 0
        assert safe(None, "fallback") == "fallback"
        assert safe(None, []) == []

    def test_works_with_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert safe(df, pd.DataFrame()) is df
        empty = safe(None, pd.DataFrame())
        assert isinstance(empty, pd.DataFrame)
        assert empty.empty

    def test_works_with_falsy_values_that_are_not_none(self):
        """0, False, '' are not None — should be returned."""
        assert safe(0, 99) == 0
        assert safe(False, True) is False
        assert safe("", "fallback") == ""


class TestSafeDf:
    """_safe_df(df) — DataFrame-specific wrapper."""

    def test_returns_df_when_not_none(self):
        df = pd.DataFrame({"x": [1]})
        result = _safe_df(df)
        assert result is df

    def test_returns_empty_df_when_none(self):
        result = _safe_df(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
