import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


# TODO: Write a simple Winsorizer transformer which
# takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float, upper_quantile: float) -> None:
        """Scikit-learn compatible transformer that performs
        Winsorization on numeric data. It limits extreme outliers
        and replaces them with the lower or upper quantile values.

        Args:
            lower_quantile: float
            The lower threshold for the data. Values below lower
            quantile will be set equal to lower_quantile

            upper_quantile: float
            The lower threshold for the data. Values below lower
            quantile will be set equal to lower_quantile.
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "Winsorizer":
        """Compute the lower and upper quantile values for winsorization.

        Args:
            X: array-like
            The input data to perform quantiles.
            y: None
            This patameter exists for compatibility with scikit-learn.

        Returns:
            self: Winsorizer
            Fitted transformer with 'lower_quantile_ and upper_quantile_
        """
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Data transformer which replaces values below/above defined quartiles
        with the upper/lower quantile

        Args:
            X: array-like
            The input data to transform

        Returns:
            X_transformed: array-like
            The winsorized version of X
        """
        X_transformed = X.copy()
        X_transformed = np.where(
            X_transformed < self.lower_quantile_,
            self.lower_quantile_,
            X_transformed,
        )
        X_transformed = np.where(
            X_transformed > self.upper_quantile_,
            self.upper_quantile_,
            X_transformed,
        )
        return X_transformed
