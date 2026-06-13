import pandas as pd
import os


REQUIRED_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

FEATURE_COLUMNS = [col for col in REQUIRED_COLUMNS if col != 'Outcome']
TARGET_COLUMN = 'Outcome'


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    _validate_columns(df)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def get_basic_info(df: pd.DataFrame) -> dict:
    counts = df[TARGET_COLUMN].value_counts()
    return {
        'n_rows': df.shape[0],
        'n_cols': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'class_distribution': counts.to_dict(),
        'class_balance_ratio': round(counts[0] / counts[1], 2)
    }
