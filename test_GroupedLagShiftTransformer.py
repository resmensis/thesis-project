import pandas as pd
import numpy as np
from transformers import GroupedLagShiftTransformer



X = pd.DataFrame(
    {
        "group_id": ["A", "A", "A", "B", "B"],
        "date": pd.to_datetime(
            [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-01-01",
                "2020-02-01",
            ]
        ),
        "return": [0.10, 0.20, 0.30, 0.40, 0.50],
        "size": [100, 110, 120, 200, 210],
    }
)

X = X.sort_values(["group_id", "date"]).reset_index(drop=True)

transformer = GroupedLagShiftTransformer(
    lag=1,
    group_col="group_id",
    value_cols=["return", "size"],
    create_new_cols=True,
)

X_lagged = transformer.fit_transform(X)

print(X)
print(X_lagged)