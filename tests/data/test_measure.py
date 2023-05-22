from __future__ import annotations

import numpy as np
import pandas as pd

from byop.data.measure import byte_size


def test_bytesize():
    # Default 8 bytes per number
    empty_df = pd.DataFrame(index=np.arange(100))
    assert byte_size(empty_df) == 800

    empty_series = pd.Series(index=np.arange(100))
    assert byte_size(empty_series) == 1600  # Empty series has to fill with NaN

    ndarray = np.arange(100)
    assert byte_size(ndarray) == 800

    assert byte_size([empty_df, empty_series, ndarray]) == 3200
