import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the healthcare dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    return df


def get_target_label(df: pd.DataFrame, target_col: str = "Medical Condition") -> np.ndarray:
    """Build a binary target array for cancer detection."""
    return (df[target_col].astype(str).str.strip().str.lower() == "cancer").astype(int).to_numpy()


def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical features."""
    numeric_features = ["Age", "Billing Amount"]
    categorical_features = [
        "Gender",
        "Blood Type",
        "Admission Type",
        "Medication",
        "Test Results",
    ]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def preprocess_features(df: pd.DataFrame, preprocessor: ColumnTransformer, fit: bool = True):
    """Preprocess the feature columns using the supplied transformer."""
    raw_features = df[
        [
            "Age",
            "Billing Amount",
            "Gender",
            "Blood Type",
            "Admission Type",
            "Medication",
            "Test Results",
        ]
    ].copy()

    raw_features.fillna("Unknown", inplace=True)
    raw_features["Gender"] = raw_features["Gender"].astype(str).str.strip().str.title()
    raw_features["Blood Type"] = raw_features["Blood Type"].astype(str).str.strip().str.upper()
    raw_features["Admission Type"] = raw_features["Admission Type"].astype(str).str.strip().str.title()
    raw_features["Test Results"] = raw_features["Test Results"].astype(str).str.strip().str.title()

    if fit:
        X = preprocessor.fit_transform(raw_features)
    else:
        X = preprocessor.transform(raw_features)

    return X
