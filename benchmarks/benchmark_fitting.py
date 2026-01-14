"""Benchmark fitting performance."""

import time

from transitkit.core.models import TransitModelJAX


def main() -> None:
    model = TransitModelJAX()
    params = {
        "t0": 0.0,
        "period": 5.0,
        "rp_over_rs": 0.1,
        "a_over_rs": 15.0,
        "inc": 89.0,
    }
    time_array = [i * 0.01 for i in range(10000)]
    start = time.time()
    model.compute(time_array, params)
    elapsed = time.time() - start
    print(f"Transit compute: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
