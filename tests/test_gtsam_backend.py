from __future__ import annotations

import sys

import pytest

from outdoor_slpf.smoothing import GtsamFixedLagSmoother, create_pose_backend, require_gtsam


def test_default_pose_backend_does_not_require_gtsam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gtsam", None)

    backend = create_pose_backend("alpha")

    assert backend.name == "alpha"


def test_gtsam_backend_reports_install_command_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gtsam", None)

    with pytest.raises(RuntimeError, match='python -m pip install "gtsam==4.2.1"'):
        require_gtsam()


def test_gtsam_backend_does_not_silently_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenBackend(GtsamFixedLagSmoother):
        def __init__(self):
            pass

        def _update_with_gtsam(self, **kwargs):
            raise RuntimeError("optimizer failed")

    backend = BrokenBackend()

    with pytest.raises(RuntimeError, match="optimizer failed"):
        backend.update(raw_pose=[0.0, 0.0, 0.0])
