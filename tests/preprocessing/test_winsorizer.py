import numpy as np
import pytest

from ps3.preprocessing import Winsorizer


# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile",
    [(0, 1), (0.05, 0.95), (0.5, 0.5)],
    # [0,1] has no winsorization
    # [0.05, 0.95] trims the bottom and top 5%
    # [0.5,0.5] trims bottom and top 50%, meaning it only leaves the median
)  # type: ignore
def test_winsorizer(lower_quantile: float, upper_quantile: float) -> None:

    X = np.random.normal(0, 1, 1000)

    wind_data = Winsorizer(lower_quantile, upper_quantile)
    wind_data.fit(X)
    X_winsorized = wind_data.transform(X)

    # test if the attributes exist after the fit step
    assert hasattr(wind_data, "lower_quantile_")
    assert hasattr(wind_data, "upper_quantile_")

    # test the correctness of fit values
    assert wind_data.lower_quantile_ == np.quantile(X, lower_quantile)
    assert wind_data.upper_quantile_ == np.quantile(X, upper_quantile)

    # test if all values are lower than upper_quantile_
    # and higher than lower_quantile_
    assert np.all(X_winsorized >= wind_data.lower_quantile_)
    assert np.all(X_winsorized <= wind_data.upper_quantile_)
