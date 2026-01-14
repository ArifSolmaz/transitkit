"""Benchmark loading performance."""

import time

from transitkit.data.pipeline import LightCurveData


def main() -> None:
    start = time.time()
    light_curve = LightCurveData(
        time=[0.0, 1.0, 2.0],
        flux=[1.0, 0.99, 1.0],
        flux_err=[0.01, 0.01, 0.01],
        quality=[0, 0, 0],
        cadence=1.0,
        mission="BENCH",
        target_id="BENCH-001",
    )
    elapsed = time.time() - start
    print(f"LightCurveData init: {elapsed:.6f}s ({len(light_curve.time)} points)")


if __name__ == "__main__":
    main()
