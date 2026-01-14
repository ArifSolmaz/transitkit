import numpy as np

from transitkit.data.pipeline import LightCurveData


def test_light_curve_data_coerces_arrays():
    light_curve = LightCurveData(
        time=[0.0, 1.0, 2.0],
        flux=[1.0, 0.99, 1.0],
        flux_err=[0.01, 0.01, 0.01],
        quality=[0, 0, 0],
        cadence=1.0,
        mission="TEST",
        target_id="TEST-001",
    )

    assert isinstance(light_curve.time, np.ndarray)
    assert light_curve.time.shape == (3,)
