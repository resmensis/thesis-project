import pandas as pd
import numpy as np
from transformers import GroupedMedianImputer

df = pd.DataFrame({
    "permno": [1, 2, 3, 4, 5],
    "year": [2020, 2020, 2020, 2021, 2021],
    "industry": ["A", "A", "B", "A", "B"],
    "bm": [0.4, np.nan, 0.8, np.nan, 1.0],
    "mom": [0.2, 0.4, np.nan, 0.6, np.nan],
})


imputer = GroupedMedianImputer(
    group_cols=["year"],
    value_cols=["bm", "mom"],
    fallback="global_median",
)

df_imputed = imputer.fit_transform(df)

print(df)
print(df_imputed)
print(imputer.group_medians_)
print(imputer.global_medians_)