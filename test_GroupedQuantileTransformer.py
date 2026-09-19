import pandas as pd
import numpy as np

from transformers import GroupedQuantileTransformer

df = pd.DataFrame({
    "permno": [1, 1, 2, 2, 1, 2],
    "date": pd.to_datetime([
        "2020-01-01",
        "2020-02-01",
        "2020-01-01",
        "2020-02-01",
        "2020-01-01",
        "2020-02-01",
    ]),
    "size": [100, 200, 1000, 2000, np.nan, np.nan],
    "bm": [0.4, 0.6, 1.2, 1.5, np.nan, np.nan],
})

list = ["size", "bm"]

transformer = GroupedQuantileTransformer(
    group_col="date",
    value_cols=list,
    output_range=(-1, 1),
    n_quantiles=100,
)

result = transformer.fit_transform(df)

print(df)
print(result)