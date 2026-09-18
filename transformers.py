from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


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





class GroupedLagShiftTransformer111111111111111(BaseEstimator, TransformerMixin):
    def _init_(self, lag=1, group_col="group_id", value_cols=None):
        self.lag = lag
        self.group_col = group_col
        self.value_cols = value_cols

    def fit(self, X, y=None):
        self.value_cols_ = self.value_cols or [c for c in X.columns if c != self.group_col]
        return self

    def transform(self, X):
        X = X.copy()
        shifted = X.groupby(self.group_col)[self.value_cols_].shift(self.lag)
        shifted.columns = [f"{c}_lag{self.lag}" for c in shifted.columns]
        return pd.concat([X.drop(columns=self.value_cols_), shifted], axis=1)


    from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GroupedLagShiftTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a grouped pandas shift to selected columns.

    Parameters
    ----------
    lag : int, default=1
        Number of rows by which to shift within each group.

        Positive values create backward lags:
            groupby(...).shift(1)

        Negative values create forward shifts:
            groupby(...).shift(-1)

    group_col : str, default="group_id"
        Column used to define groups.

    value_cols : sequence of str or None, default=None
        Columns to shift. If None, all columns except group_col are shifted.

    create_new_cols : bool, default=True
        If True, preserve the original value columns and append shifted columns.
        If False, replace the original value columns with shifted values.

    suffix : str or None, default=None
        Suffix used for newly created columns. If None, a suffix is generated
        from lag, for example "_lag1" or "_shift_forward1".
    """

    def __init__(
        self,
        lag: int = 1,
        group_col: str = "group_id",
        value_cols: Sequence[str] | None = None,
        create_new_cols: bool = True,
        suffix: str | None = None,
    ):
        self.lag = lag
        self.group_col = group_col
        self.value_cols = value_cols
        self.create_new_cols = create_new_cols
        self.suffix = suffix

    def fit(self, X: pd.DataFrame, y=None):
        """Validate the input and learn the columns used during transformation."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not isinstance(self.lag, Integral):
            raise TypeError("lag must be an integer.")

        if not isinstance(self.group_col, str):
            raise TypeError("group_col must be a string.")

        if not isinstance(self.create_new_cols, bool):
            raise TypeError("create_new_cols must be a boolean.")

        if self.suffix is not None and not isinstance(self.suffix, str):
            raise TypeError("suffix must be a string or None.")

        if self.group_col not in X.columns:
            raise ValueError(
                f"group_col={self.group_col!r} is not present in X."
            )

        if self.value_cols is None:
            value_cols = [
                column for column in X.columns
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
                suffix = f"_lag{self.lag}"
            else:
                suffix = f"_shift_forward{abs(self.lag)}"
        else:
            suffix = self.suffix

        self.suffix_ = suffix

        self.shifted_cols_ = [
            f"{column}{self.suffix_}"
            for column in self.value_cols_
        ]

        if len(set(self.shifted_cols_)) != len(self.shifted_cols_):
            raise ValueError("Generated shifted column names are not unique.")

        if self.create_new_cols:
            collisions = [
                column for column in self.shifted_cols_
                if column in X.columns
            ]

            if collisions:
                raise ValueError(
                    "Generated shifted columns already exist in X: "
                    f"{collisions}"
                )

            self.feature_names_out_ = np.asarray(
                list(X.columns) + self.shifted_cols_,
                dtype=object,
            )
        else:
            self.feature_names_out_ = np.asarray(
                [
                    self.group_col if column == self.group_col
                    else (
                        column
                        if column not in self.value_cols_
                        else f"{column}{self.suffix_}"
                    )
                    for column in X.columns
                ],
                dtype=object,
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Shift value columns independently within each group."""
        check_is_fitted(
            self,
            attributes=[
                "value_cols_",
                "shifted_cols_",
                "feature_names_out_",
            ],
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        missing_cols = [
            column for column in self.feature_names_in_
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

        if self.create_new_cols:
            shifted = shifted.copy()
            shifted.columns = self.shifted_cols_

            return pd.concat(
                [X.copy(), shifted],
                axis=1,
            )

        X_out = X.copy()
        X_out.loc[:, self.value_cols_] = shifted.to_numpy()

        return X_out

    def get_feature_names_out(
        self,
        input_features=None,
    ) -> np.ndarray:
        """Return output column names for sklearn compatibility."""
        check_is_fitted(self, attributes=["feature_names_out_"])

        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)

            if not np.array_equal(
                input_features,
                self.feature_names_in_,
            ):
                raise ValueError(
                    "input_features do not match the columns seen during fit."
                )

        return self.feature_names_out_.copy()