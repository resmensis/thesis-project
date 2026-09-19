from __future__ import annotations

import numpy as np
import pandas as pd

from numbers import Integral
from typing import Sequence
from collections.abc import Hashable
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import QuantileTransformer


class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing numeric values using group-specific medians.

    Parameters
    ----------
    group_cols : str or list of str
        Columns defining the groups.

    value_cols : str, list of str, or None
        Numeric columns to impute. If None, all columns not in group_cols
        are used.

    fallback : {"global_median", "constant", "leave_missing"}
        Behavior when a group-specific median is unavailable.

    fill_value : float or None
        Value used when fallback="constant".
    """

    def __init__(
        self,
        group_cols,
        value_cols=None,
        fallback="global_median",
        fill_value=0.0,
    ):
        self.group_cols = group_cols
        self.value_cols = value_cols
        self.fallback = fallback
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self._validate_parameters()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self.group_cols_ = self._as_list(self.group_cols)
        self._check_columns_exist(X, self.group_cols_, "group")

        if self.value_cols is None:
            self.value_cols_ = [
                col for col in X.columns
                if col not in self.group_cols_
            ]
        else:
            self.value_cols_ = self._as_list(self.value_cols)

        self._check_columns_exist(X, self.value_cols_, "value")

        non_numeric = X[self.value_cols_].select_dtypes(
            exclude=np.number
        ).columns.tolist()

        if non_numeric:
            raise TypeError(
                "value_cols must be numeric. "
                f"Non-numeric columns: {non_numeric}"
            )

        # One row per group, one column per imputable feature.
        self.group_medians_ = (
            X.groupby(
                self.group_cols_,
                dropna=False,
                sort=False,
                observed=False,
            )[self.value_cols_]
            .median()
        )

        # Used for unseen groups or groups whose median is NaN.
        self.global_medians_ = X[self.value_cols_].median()

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        check_is_fitted(
            self,
            attributes=[
                "group_medians_",
                "global_medians_",
                "feature_names_in_",
            ],
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        self._check_same_columns(X)

        result = X.copy()

        # Align learned group medians with the rows being transformed.
        """ 
        row_group_medians = self.group_medians_.reindex(
            pd.MultiIndex.from_frame(X[self.group_cols_])
        )
        """
        

        row_group_medians = self.group_medians_.reindex(
            X[self.group_cols_[0]]
            if len(self.group_cols_) == 1
            else pd.MultiIndex.from_frame(X[self.group_cols_])
        )

        row_group_medians.index = X.index

        for column in self.value_cols_:
            missing = result[column].isna()

            replacements = row_group_medians[column]

            if self.fallback == "global_median":
                replacements = replacements.fillna(
                    self.global_medians_[column]
                )
            elif self.fallback == "constant":
                replacements = replacements.fillna(self.fill_value)
            elif self.fallback == "leave_missing":
                pass

            result.loc[missing, column] = replacements.loc[missing]

        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")

        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)

            if not np.array_equal(
                input_features,
                self.feature_names_in_,
            ):
                raise ValueError(
                    "input_features does not match feature_names_in_."
                )

        return self.feature_names_in_.copy()

    @staticmethod
    def _as_list(columns):
        if isinstance(columns, str):
            return [columns]
        return list(columns)

    @staticmethod
    def _check_columns_exist(X, columns, column_type):
        missing = [column for column in columns if column not in X.columns]

        if missing:
            raise ValueError(
                f"Unknown {column_type} columns: {missing}"
            )

    def _check_same_columns(self, X):
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                "X must have the same columns and column order as during fit."
            )

    def _validate_parameters(self):
        valid_fallbacks = {
            "global_median",
            "constant",
            "leave_missing",
        }

        if self.fallback not in valid_fallbacks:
            raise ValueError(
                f"fallback must be one of {valid_fallbacks}, "
                f"got {self.fallback!r}."
            )



class GroupedLagShiftTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a grouped shift to selected pandas DataFrame columns.

    Parameters
    ----------
    lag : int, default=1
        Shift length.

        Positive values create backward lags:
            groupby(...).shift(1)

        Negative values create forward shifts:
            groupby(...).shift(-1)

    group_col : str, default="group_id"
        Column used to define groups.

    value_cols : sequence of str or None, default=None
        Columns to shift.S If None, all columns except group_col are shifted.

    create_new_cols : bool, default=True
        If True, preserve the original columns and add shifted columns.

        If False, replace the original value columns with shifted values.

    rename_mode : {"lagged", "original", "none"}, default="lagged"
        Determines how columns are named when create_new_cols=True.

        "lagged":
            Original column stays unchanged.
            New column receives the suffix.

            return -> return, return_lag1

        "original":
            New lagged column keeps the original name.
            Original column receives "_original".

            return -> return_original, return

        "none":
            No renaming is performed. This option is mainly useful when
            create_new_cols=False.

    suffix : str or None, default=None
        Suffix for the lagged column. If None, a suffix is generated from lag.

    original_suffix : str, default="_original"
        Suffix added to the original column when rename_mode="original".
    """

    def __init__(
        self,
        lag: int = 1,
        group_col: str = "group_id",
        value_cols: Sequence[str] | None = None,
        create_new_cols: bool = True,
        rename_mode: str = "lagged",
        suffix: str | None = None,
        original_suffix: str = "_original",
    ):
        self.lag = lag
        self.group_col = group_col
        self.value_cols = value_cols
        self.create_new_cols = create_new_cols
        self.rename_mode = rename_mode
        self.suffix = suffix
        self.original_suffix = original_suffix

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not isinstance(self.lag, Integral):
            raise TypeError("lag must be an integer.")

        if not isinstance(self.group_col, str):
            raise TypeError("group_col must be a string.")

        if not isinstance(self.create_new_cols, bool):
            raise TypeError("create_new_cols must be a boolean.")

        valid_rename_modes = {"lagged", "original", "none"}

        if self.rename_mode not in valid_rename_modes:
            raise ValueError(
                f"rename_mode must be one of {valid_rename_modes}, "
                f"got {self.rename_mode!r}."
            )

        if not isinstance(self.original_suffix, str):
            raise TypeError("original_suffix must be a string.")

        if self.group_col not in X.columns:
            raise ValueError(
                f"group_col={self.group_col!r} is not present in X."
            )

        if self.value_cols is None:
            value_cols = [
                column
                for column in X.columns
                if column != self.group_col
            ]
        else:
            value_cols = list(self.value_cols)

            missing_cols = [
                column for column in value_cols
                if column not in X.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"The following value_cols are missing from X: "
                    f"{missing_cols}"
                )

            if self.group_col in value_cols:
                raise ValueError(
                    "group_col must not be included in value_cols."
                )

        if not value_cols:
            raise ValueError("At least one value column is required.")

        self.value_cols_ = value_cols
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        if self.suffix is None:
            if self.lag >= 0:
                self.suffix_ = f"_lag{self.lag}"
            else:
                self.suffix_ = f"_shift_forward{abs(self.lag)}"
        else:
            self.suffix_ = self.suffix

        self.lagged_names_ = [
            f"{column}{self.suffix_}"
            for column in self.value_cols_
        ]

        self.original_names_ = [
            f"{column}{self.original_suffix}"
            for column in self.value_cols_
        ]

        self._build_feature_names(X)
        return self

    def _build_feature_names(self, X: pd.DataFrame):
        input_columns = list(X.columns)

        if not self.create_new_cols:
            self.feature_names_out_ = np.asarray(
                input_columns,
                dtype=object,
            )
            return

        if self.rename_mode == "lagged":
            output_columns = input_columns + self.lagged_names_

        elif self.rename_mode == "original":
            original_renames = dict(
                zip(self.value_cols_, self.original_names_)
            )

            original_columns = [
                original_renames.get(column, column)
                for column in input_columns
            ]

            output_columns = original_columns + self.value_cols_

        else:
            output_columns = input_columns + self.value_cols_

        if len(set(output_columns)) != len(output_columns):
            raise ValueError(
                "The transformer would create duplicate output column names: "
                f"{output_columns}"
            )

        self.feature_names_out_ = np.asarray(
            output_columns,
            dtype=object,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=[
                "value_cols_",
                "lagged_names_",
                "original_names_",
                "feature_names_out_",
            ],
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        missing_cols = [
            column
            for column in self.feature_names_in_
            if column not in X.columns
        ]

        if missing_cols:
            raise ValueError(
                f"X is missing columns seen during fit: {missing_cols}"
            )

        shifted = (
            X.groupby(
                self.group_col,
                sort=False,
                dropna=False,
            )[self.value_cols_]
            .shift(self.lag)
        )

        if not self.create_new_cols:
            X_out = X.copy()
            X_out.loc[:, self.value_cols_] = shifted.to_numpy()
            return X_out

        if self.rename_mode == "lagged":
            shifted = shifted.copy()
            shifted.columns = self.lagged_names_

            collisions = [
                column
                for column in shifted.columns
                if column in X.columns
            ]

            if collisions:
                raise ValueError(
                    f"Lagged columns already exist in X: {collisions}"
                )

            return pd.concat(
                [X.copy(), shifted],
                axis=1,
            )

        if self.rename_mode == "original":
            X_out = X.rename(
                columns=dict(
                    zip(self.value_cols_, self.original_names_)
                )
            )

            shifted = shifted.copy()
            shifted.columns = self.value_cols_

            return pd.concat(
                [X_out, shifted],
                axis=1,
            )

        shifted = shifted.copy()
        shifted.columns = self.value_cols_

        collisions = [
            column
            for column in shifted.columns
            if column in X.columns
        ]

        if collisions:
            raise ValueError(
                "rename_mode='none' would create duplicate columns: "
                f"{collisions}"
            )

        return pd.concat(
            [X.copy(), shifted],
            axis=1,
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_names_out_"])

        if input_features is not None:
            input_features = np.asarray(
                input_features,
                dtype=object,
            )

            if not np.array_equal(
                input_features,
                self.feature_names_in_,
            ):
                raise ValueError(
                    "input_features do not match the columns seen during fit."
                )

        return self.feature_names_out_.copy()


class GroupedQuantileTransformer(BaseEstimator, TransformerMixin):
    """Apply one sklearn QuantileTransformer per group to selected columns.

    Parameters
    ----------
    group_col : str or int
        Group column in a DataFrame, or group-column index for array-like input.
    value_cols : list[str] or list[int] or None, default=None
        Columns to transform. For DataFrames, use column names. For array-like
        input, use column indices in the original input. If None, all columns
        except group_col are transformed.
    output_range : tuple[float, float], default=(-1.0, 1.0)
        Range of the transformed uniform output.
    n_quantiles : int, default=1000
        Passed to sklearn.preprocessing.QuantileTransformer.
    subsample : int or None, default=10000
        Passed to QuantileTransformer. ``None`` requires a compatible sklearn
        version.
    random_state : int or None, default=None
        Passed to QuantileTransformer.
    copy : bool, default=True
        Whether to copy input data before transforming.
    """

    def __init__(
        self,
        group_col: str | int,
        value_cols: list[str] | None = None,
        output_range: tuple[float, float] = (-1.0, 1.0),
        n_quantiles: int = 1000,
        subsample: int | None = 10000,
        random_state: int | None = None,
        copy: bool = True,
    ) -> None:
        self.group_col = group_col
        self.value_cols = value_cols
        self.output_range = output_range
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def _validate_params(self) -> None:
        if len(self.output_range) != 2 or self.output_range[0] >= self.output_range[1]:
            raise ValueError("output_range must be a pair (low, high) with low < high")
        if self.n_quantiles < 1:
            raise ValueError("n_quantiles must be at least 1")

    def _resolve_columns(self, X: Any, fitting: bool = False):
        if isinstance(X, pd.DataFrame):
            if not isinstance(self.group_col, str):
                raise TypeError("group_col must be a column name for DataFrame input")
            if self.group_col not in X.columns:
                raise ValueError(f"Unknown group column: {self.group_col!r}")

            if self.value_cols is None:
                value_cols = [col for col in X.columns if col != self.group_col]
            else:
                value_cols = list(self.value_cols)
                missing = [col for col in value_cols if col not in X.columns]
                if missing:
                    raise ValueError(f"Unknown value columns: {missing}")
                if self.group_col in value_cols:
                    raise ValueError("group_col cannot also be in value_cols")

            if fitting:
                self.value_cols_ = np.asarray(value_cols, dtype=object)
            elif list(value_cols) != list(self.value_cols_):
                raise ValueError("Input value columns differ from those used during fit")

            return X[self.group_col], X[value_cols], value_cols, True

        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a 2-dimensional DataFrame or array-like object")
        if not isinstance(self.group_col, int):
            raise TypeError("group_col must be an integer for array-like input")
        if not 0 <= self.group_col < array.shape[1]:
            raise ValueError("group_col is outside the input columns")

        if self.value_cols is None:
            value_cols = [i for i in range(array.shape[1]) if i != self.group_col]
        else:
            value_cols = list(self.value_cols)
            if any(not isinstance(i, int) for i in value_cols):
                raise TypeError("value_cols must contain integer indices for array-like input")
            if any(i < 0 or i >= array.shape[1] for i in value_cols):
                raise ValueError("value_cols contains an index outside the input columns")
            if self.group_col in value_cols:
                raise ValueError("group_col cannot also be in value_cols")

        if fitting:
            self.value_cols_ = np.asarray(value_cols, dtype=int)
        elif not np.array_equal(value_cols, self.value_cols_):
            raise ValueError("Input value columns differ from those used during fit")

        return array[:, self.group_col], array[:, value_cols], value_cols, False

    def fit(self, X: Any, y: Any = None):
        self._validate_params()
        groups, values, value_cols, is_dataframe = self._resolve_columns(X, fitting=True)
        numeric = values.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float) if is_dataframe else np.asarray(values, dtype=float)

        self.n_features_in_ = numeric.shape[1]
        self.groups_ = pd.unique(groups)
        self.group_transformers_: dict[Hashable, list[QuantileTransformer | None]] = {}
        self._is_dataframe_ = is_dataframe

        for group in self.groups_:
            group_mask = np.asarray(groups == group)
            transformers = []
            for j in range(self.n_features_in_):
                observed = numeric[group_mask, j]
                observed = observed[~np.isnan(observed)]
                if observed.size == 0:
                    transformers.append(None)
                    continue

                kwargs = {
                    "n_quantiles": min(self.n_quantiles, observed.size),
                    "output_distribution": "uniform",
                    "random_state": self.random_state,
                    "copy": self.copy,
                }
                if self.subsample is not None:
                    kwargs["subsample"] = self.subsample

                transformer = QuantileTransformer(**kwargs)
                transformer.fit(observed.reshape(-1, 1))
                transformers.append(transformer)

            self.group_transformers_[group] = transformers

        return self

    def transform(self, X: Any):
        check_is_fitted(self, ["group_transformers_", "value_cols_"])
        groups, values, value_cols, is_dataframe = self._resolve_columns(X, fitting=False)
        numeric = values.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float) if is_dataframe else np.asarray(values, dtype=float)

        result = X.copy() if self.copy else X
        low, high = self.output_range

        for group in pd.unique(groups):
            if group not in self.group_transformers_:
                raise ValueError(f"Unknown group {group!r} encountered during transform")

            group_mask = np.asarray(groups == group)
            for j, transformer in enumerate(self.group_transformers_[group]):
                valid = group_mask & ~np.isnan(numeric[:, j])
                if transformer is None or not valid.any():
                    continue

                uniform = transformer.transform(numeric[valid, j].reshape(-1, 1)).ravel()
                result_values = low + (high - low) * uniform

                if is_dataframe:
                    result.loc[result.index[valid], value_cols[j]] = result_values
                else:
                    result[valid, value_cols[j]] = result_values

        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["value_cols_"])
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self._is_dataframe_:
            return np.asarray(self.feature_names_in_, dtype=object) if hasattr(self, "feature_names_in_") else self.value_cols_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
