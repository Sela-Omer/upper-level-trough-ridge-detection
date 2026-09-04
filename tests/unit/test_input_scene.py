from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from ultr_detection.config import Task
from ultr_detection.inference import load_scene


def test_load_scene_accepts_documented_netcdf_schema(tmp_path: Path) -> None:
    path = tmp_path / "scene.nc"
    with netCDF4.Dataset(path, "w") as target:
        target.createDimension("latitude", 2)
        target.createDimension("longitude", 3)
        target.createVariable("latitude", "f4", ("latitude",))[:] = [40, 39]
        target.createVariable("longitude", "f4", ("longitude",))[:] = [10, 11, 12]
        for name, value in (("z500", 5500), ("u500", 10), ("v500", -2)):
            target.createVariable(name, "f4", ("latitude", "longitude"))[:] = value

    scene = load_scene(path, task=Task.RIDGE, month_index=7)

    assert scene.task is Task.RIDGE
    assert scene.month_index == 7
    assert scene.z500.shape == (2, 3)
    assert np.allclose(scene.longitude, [10, 11, 12])
