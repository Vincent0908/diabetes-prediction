import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


# Columns where 0 is biologically impossible; treated as missing
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def replace_impossible_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zero values in clinical columns with column median."""
    df = df.copy()
    for col in ZERO_AS_MISSING:
        if col in df.columns:
            n_zeros = (df[col] == 0).sum()
            if n_zeros > 0:
                median_val = df[col].replace(0, np.nan).median()
                df[col] = df[col].replace(0, median_val)
                print(f"  [CLEAN] {col}: {n_zeros} zeros -> median ({median_val:.2f})")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add three domain-informed derived features."""
    df = df.copy()

    # Clinical BMI categories: underweight/normal/overweight/obese
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25.0, 30.0, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Proxy for insulin resistance
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)

    # Interaction term
    df['Age_Pregnancies'] = df['Age'] * df['Pregnancies']

    print(f"  [FEAT] 3 features added. Total: {df.shape[1] - 1}")
    return df


def split_features_target(df: pd.DataFrame, target: str = 'Outcome'):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_features(X_train, X_test, scaler_path: str = None):
    """Fit scaler on train only to avoid data leakage."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"  [SAVE] Scaler -> {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing_pipeline(df: pd.DataFrame, models_dir: str = 'models') -> dict:
    print("\n[PREPROCESSING]")

    df_clean = replace_impossible_zeros(df)
    df_feat = add_engineered_features(df_clean)

    X, y = split_features_target(df_feat)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"  [SPLIT] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scaler_path)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler,
        'df_processed': df_feat
    }
