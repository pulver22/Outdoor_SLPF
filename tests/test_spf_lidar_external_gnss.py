from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np


def _install_spf_lidar_import_stubs() -> None:
    class _DummyYOLO:
        def __init__(self, *_args, **_kwargs):
            pass

        def to(self, *_args, **_kwargs):
            return self

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda *_args: None, synchronize=lambda: None),
        manual_seed=lambda *_args: None,
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)),
    )
    sys.modules.setdefault("cv2", types.SimpleNamespace())
    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("geopandas", types.SimpleNamespace())
    sys.modules.setdefault("pandas", types.SimpleNamespace(notna=lambda value: value is not None))
    sys.modules.setdefault("pyproj", types.SimpleNamespace(Transformer=types.SimpleNamespace(from_crs=lambda *_args, **_kwargs: None)))
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable, **_kwargs: iterable))
    sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=_DummyYOLO))


def _import_spf_lidar():
    _install_spf_lidar_import_stubs()
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("scripts.spf_lidar")


def test_external_noisy_gnss_tum_loader_and_interpolator_preserve_outages(tmp_path: Path) -> None:
    spf_lidar = _import_spf_lidar()
    tum = tmp_path / "degraded.tum"
    tum.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
        "1.0 2.0 4.0 0.0 0.0 0.0 0.0 1.0\n"
        "2.0 nan nan 0.0 0.0 0.0 0.0 1.0\n",
        encoding="utf-8",
    )

    degraded = spf_lidar.load_external_noisy_gnss_tum(tum)

    assert np.allclose(spf_lidar.interpolate_external_noisy_gnss(degraded, 0.5), (1.0, 2.0))
    outage_x, outage_y = spf_lidar.interpolate_external_noisy_gnss(degraded, 1.5)
    assert np.isnan(outage_x)
    assert np.isnan(outage_y)
    assert spf_lidar.first_finite_external_gnss_xy(degraded) == (0.0, 0.0)


def test_external_noisy_gnss_interpolator_returns_nan_outside_time_range() -> None:
    spf_lidar = _import_spf_lidar()
    degraded = np.asarray([[10.0, 1.0, 2.0, 0.0], [11.0, 3.0, 4.0, 0.0]], dtype=np.float64)

    x, y = spf_lidar.interpolate_external_noisy_gnss(degraded, 9.5)

    assert np.isnan(x)
    assert np.isnan(y)
