"""
transformers.py
---------------
Custom sklearn-compatible transformers. Each transformer follows the
fit/transform/fit_transform contract so they plug directly into a
sklearn Pipeline.

KEY CONCEPT — Why custom transformers instead of ad-hoc mutations?
  If you write df['col'] = encoder.fit_transform(df['col']) outside a
  pipeline, you'll accidentally fit on the full dataset before splitting,
  leaking information from the validation fold into training.

  Inside a Pipeline, sklearn guarantees that only .fit() is called on
  training data, and only .transform() is called on validation/test data.
  This is the correct way to prevent data leakage.
"""

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_polars(X) -> pl.DataFrame:
    """
    Convert input data to a Polars DataFrame.

    This ensures compatibility with sklearn pipelines, which may pass
    numpy arrays between steps (e.g. during cross-validation), while
    user-facing inputs are typically pandas or Polars DataFrames.

    :param X: Input data (Polars, pandas, or numpy)
    :return: Polars DataFrame
    :raises TypeError: If input cannot be converted
    """
    if isinstance(X, pl.DataFrame):
        return X

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pl.from_numpy(X)

    try:
        return pl.from_pandas(X)
    except Exception:
        raise TypeError(f"Cannot convert {type(X)} to Polars DataFrame")

def _col_names(df: pl.DataFrame, stored_names: Optional[List[str]]) -> List[str]:
    """
    Retrieve column names, restoring original names if they were lost.

    When sklearn passes numpy arrays between pipeline steps, Polars assigns
    generic column names ('column_0', ...). If original names are available,
    they are used instead.

    :param df: Polars DataFrame
    :param stored_names: Previously stored column names
    :return: List of column names
    """
    if stored_names and all(c.startswith("column_") for c in df.columns):
        return stored_names
    return df.columns


# ---------------------------------------------------------------------------
# 1. TargetEncoder
# ---------------------------------------------------------------------------

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target encoding for categorical features.

    Each category is replaced with a weighted average of its target mean
    and the global target mean to reduce overfitting on rare categories.

    :param cols: List of categorical columns to encode
    :param smoothing: Strength of smoothing toward global mean
    """

    def __init__(self, cols: List[str], smoothing: float = 10.0):
        self.cols = cols
        self.smoothing = smoothing

    def fit(self, X, y):
        """
        Learn encoding mappings for each categorical column.

        :param X: Feature matrix
        :param y: Target values
        :return: self
        """
        df = _to_polars(X)

        if not isinstance(y, pl.Series):
            y_series = pl.Series("__target__", np.asarray(y, dtype=float))
        else:
            y_series = y.alias("__target__").cast(pl.Float64)

        self.global_mean_ = float(y_series.mean())
        self.mappings_: dict[str, dict] = {}

        df = df.with_columns(y_series)

        for col in self.cols:
            agg = (
                df
                .with_columns(pl.col(col).cast(pl.Utf8).alias("__key__"))
                .group_by("__key__")
                .agg([
                    pl.col("__target__").mean().alias("cat_mean"),
                    pl.col("__target__").count().alias("n"),
                ])
                .with_columns([
                    (
                        (pl.col("n") * pl.col("cat_mean") + self.smoothing * self.global_mean_)
                        / (pl.col("n") + self.smoothing)
                    ).alias("encoded")
                ])
            )

            keys = agg["__key__"].to_list()
            values = agg["encoded"].to_list()
            self.mappings_[col] = dict(zip(keys, values))

        return self

    def transform(self, X):
        """
        Apply learned target encoding to input data.

        Unseen categories are mapped to the global mean.

        :param X: Feature matrix
        :return: Transformed numpy array (sklearn-compatible)
        """
        check_is_fitted(self, "mappings_")
        df = _to_polars(X)

        self._col_names = list(df.columns)

        result_cols = []
        for col in df.columns:
            if col in self.mappings_:
                mapping = self.mappings_[col]

                encoded = (
                    df[col]
                    .cast(pl.Utf8)
                    .replace(
                        list(mapping.keys()),
                        list(mapping.values()),
                        default=None
                    )
                    .cast(pl.Float64)
                    .fill_null(self.global_mean_)
                )
                result_cols.append(encoded.alias(col))
            else:
                result_cols.append(df[col])

        return df.select(result_cols).to_numpy()


# ---------------------------------------------------------------------------
# 2. FrequencyEncoder
# ---------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features using their frequency (proportion) in the training data.

    Each category is replaced with its relative frequency:
        freq(category) = count(category) / total_samples

    :param cols: List of categorical columns to encode
    """

    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        """
        Compute frequency mappings for each categorical column.

        :param X: Feature matrix
        :param y: Ignored (present for sklearn compatibility)
        :return: self
        """
        df = _to_polars(X)

        self.freq_maps_: dict[str, dict] = {}
        n = len(df)

        for col in self.cols:
            vc = df[col].cast(pl.Utf8).value_counts()

            keys = vc[col].to_list()
            counts = vc["count"].to_list()

            self.freq_maps_[col] = {
                k: c / n for k, c in zip(keys, counts)
            }

        return self

    def transform(self, X):
        """
        Apply frequency encoding to input data.

        Unseen categories are assigned a frequency of 0.0.

        :param X: Feature matrix
        :return: Transformed numpy array (sklearn-compatible)
        """
        check_is_fitted(self, "freq_maps_")
        df = _to_polars(X)

        result_cols = []
        for col in df.columns:
            if col in self.freq_maps_:
                mapping = self.freq_maps_[col]

                encoded = (
                    df[col]
                    .cast(pl.Utf8)
                    .replace(
                        list(mapping.keys()),
                        list(mapping.values()),
                        default=None
                    )
                    .cast(pl.Float64)
                    .fill_null(0.0)
                )
                result_cols.append(encoded.alias(col))
            else:
                result_cols.append(df[col])

        return df.select(result_cols).to_numpy()


# ---------------------------------------------------------------------------
# 3. DataFrameImputer
# ---------------------------------------------------------------------------

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in a DataFrame while preserving column names and types.

    - Numeric columns are filled using mean or median.
    - Categorical columns are filled using the most frequent value (mode).

    :param num_strategy: Strategy for numeric columns ("mean" or "median")
    :param cat_strategy: Strategy for categorical columns ("most_frequent")
    """

    def __init__(self, num_strategy: str = "median", cat_strategy: str = "most_frequent"):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy

    def fit(self, X, y=None):
        """
        Compute fill values for each column based on the chosen strategies.

        :param X: Feature matrix
        :param y: Ignored (present for sklearn compatibility)
        :return: self
        """
        df = _to_polars(X)

        self.num_cols_ = df.select(cs.numeric()).columns
        self.cat_cols_ = df.select(cs.string()).columns

        self.num_fill_: dict[str, float] = {}
        for col in self.num_cols_:
            series = df[col].drop_nulls()
            val = float(
                series.median() if self.num_strategy == "median"
                else series.mean()
            )
            self.num_fill_[col] = val

        self.cat_fill_: dict[str, str] = {}
        for col in self.cat_cols_:
            modes = df[col].drop_nulls().mode()
            self.cat_fill_[col] = modes[0] if len(modes) > 0 else ""

        return self

    def transform(self, X):
        """
        Apply imputation using values learned during fit.

        :param X: Feature matrix
        :return: Transformed Polars DataFrame
        """
        check_is_fitted(self, ["num_fill_", "cat_fill_"])
        df = _to_polars(X)

        fill_exprs = []

        for col, val in self.num_fill_.items():
            if col in df.columns:
                fill_exprs.append(pl.col(col).fill_null(val))

        for col, val in self.cat_fill_.items():
            if col in df.columns:
                fill_exprs.append(pl.col(col).fill_null(val))

        if fill_exprs:
            df = df.with_columns(fill_exprs)

        return df

# ---------------------------------------------------------------------------
# 4. RobustScalerDF
# ---------------------------------------------------------------------------

class RobustScalerDF(BaseEstimator, TransformerMixin):
    """
    Scale numeric features using the Interquartile Range (IQR).

    Each value is transformed as:
        (x - median) / IQR

    This makes the scaling robust to outliers.

    :param cols: List of columns to scale (defaults to all numeric columns)
    """

    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols

    def fit(self, X, y=None):
        """
        Compute median and IQR for each column.

        :param X: Feature matrix
        :param y: Ignored (present for sklearn compatibility)
        :return: self
        """
        df = _to_polars(X)

        cols = self.cols or df.select(cs.numeric()).columns
        self.cols_: List[str] = [c for c in cols if c in df.columns]

        self.medians_: dict[str, float] = {}
        self.iqr_: dict[str, float] = {}

        for col in self.cols_:
            s = df[col].drop_nulls()

            q25 = float(s.quantile(0.25, interpolation="midpoint"))
            q75 = float(s.quantile(0.75, interpolation="midpoint"))

            self.medians_[col] = float(s.median())

            iqr = q75 - q25
            self.iqr_[col] = iqr if iqr != 0 else 1.0

        return self

    def transform(self, X):
        """
        Apply robust scaling using statistics learned during fit.

        :param X: Feature matrix
        :return: Transformed Polars DataFrame
        """
        check_is_fitted(self, "medians_")
        df = _to_polars(X)

        scale_exprs = [
            ((pl.col(col) - self.medians_[col]) / self.iqr_[col]).alias(col)
            for col in self.cols_
            if col in df.columns
        ]

        if scale_exprs:
            df = df.with_columns(scale_exprs)

        return df

# ---------------------------------------------------------------------------
# 5. InteractionFeatures
# ---------------------------------------------------------------------------

class InteractionFeatures(BaseEstimator, TransformerMixin):
    """
    Generate pairwise interaction features by multiplying selected columns.

    For each pair (a, b), a new feature is created:
        a_x_b = a * b

    :param cols: List of columns to use for generating interactions
    """

    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        """
        Precompute all column pairs for interaction.

        :param X: Feature matrix
        :param y: Ignored (present for sklearn compatibility)
        :return: self
        """
        self.pairs_ = [
            (self.cols[i], self.cols[j])
            for i in range(len(self.cols))
            for j in range(i + 1, len(self.cols))
        ]
        return self

    def transform(self, X):
        """
        Create interaction features based on precomputed column pairs.

        :param X: Feature matrix
        :return: Transformed numpy array (sklearn-compatible)
        """
        check_is_fitted(self, "pairs_")
        df = _to_polars(X)

        interaction_exprs = [
            (pl.col(a) * pl.col(b)).alias(f"{a}_x_{b}")
            for a, b in self.pairs_
            if a in df.columns and b in df.columns
        ]

        if interaction_exprs:
            df = df.with_columns(interaction_exprs)

        return df.to_numpy()