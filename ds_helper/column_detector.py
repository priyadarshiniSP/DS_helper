import pandas as pd

def detect_column_types(df: pd.DataFrame, threshold: int = 20) -> dict:
    column_types = {
        'numerical': [],
        'categorical': [],
        'text': []
    }

    for col in df.columns:
        unique_vals = df[col].nunique(dropna=True)
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            column_types['numerical'].append(col)
        elif unique_vals < threshold:
            column_types['categorical'].append(col)
        elif pd.api.types.is_object_dtype(dtype):
            if unique_vals >= threshold:
                column_types['text'].append(col)
            else:
                column_types['categorical'].append(col)
        else:
            column_types['text'].append(col)

    return column_types