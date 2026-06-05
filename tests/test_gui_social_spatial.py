"""
Smoke tests for the Social Place Fields GUI surface.

Streamlit widgets can't render without a running session, so these cover:
* SocialSpatialParams: immutability, hashability, as_dict, cache-key distinctness.
* gui.tabs.social_spatial and gui.app import cleanly.
"""

import importlib
import sys

import pytest


class TestSocialSpatialParams:
    def _make(self, **kw):
        from gui.state import SocialSpatialParams
        base = dict(
            focal="631", targets=("631", "632", "633"),
            bin_size_cm=5.0, smoothing_sigma_cm=5.0, speed_threshold_cms=5.0,
            speed_filter_subject="target", n_shuffles=200, use_quality_cells=True,
        )
        base.update(kw)
        return SocialSpatialParams(**base)

    def test_construct_and_as_dict(self):
        d = self._make().as_dict()
        assert d["focal"] == "631"
        assert d["targets"] == ("631", "632", "633")
        assert d["speed_filter_subject"] == "target"

    def test_is_frozen_and_hashable(self):
        p = self._make()
        with pytest.raises((AttributeError, Exception)):
            p.focal = "999"  # type: ignore[misc]
        assert hash(p) == hash(p)

    def test_cache_key_distinct(self):
        from gui.cache import _make_key
        p1 = self._make()
        p2 = self._make(bin_size_cm=10.0)
        p3 = self._make(focal="632")
        p4 = self._make(targets=("631", "632"))
        keys = {
            _make_key("631", "20251216", None, p.as_dict())
            for p in (p1, p2, p3, p4)
        }
        assert len(keys) == 4


class TestImports:
    def test_tab_imports(self):
        pytest.importorskip("streamlit")
        for mod in ["gui.tabs.social_spatial", "gui.tabs", "gui.state"]:
            sys.modules.pop(mod, None)
        m = importlib.import_module("gui.tabs.social_spatial")
        assert hasattr(m, "render") and callable(m.render)

    def test_app_imports(self):
        pytest.importorskip("streamlit")
        pytest.importorskip("networkx")
        sys.modules.pop("gui.app", None)
        m = importlib.import_module("gui.app")
        assert m is not None
