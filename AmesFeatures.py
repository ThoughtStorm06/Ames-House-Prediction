import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder


# =====================================================================================
# Domain-specific config — original column names w/ spaces, exactly as in AmesHousing.csv
# =====================================================================================

IDENTIFIER_COLUMNS = ["Order", "PID"]

# Categorical columns where NaN means "feature absent" (not "unknown") — needed
# so QUALITY_MAPPINGS below has a 'None' string to map to 0.
NONE_FILL_CATEGORICAL = {
    "Alley": "NoAlley",
    "Mas Vnr Type": "None",
    "Bsmt Qual": "None",
    "Bsmt Cond": "None",
    "Bsmt Exposure": "None",
    "BsmtFin Type 1": "None",
    "BsmtFin Type 2": "None",
    "Fireplace Qu": "None",
    "Pool QC": "None",
    "Fence": "NoFence",
    "Misc Feature": "None",
    "Garage Type": "None",
    "Garage Finish": "None",
    "Garage Qual": "None",
    "Garage Cond": "None",
}

# Numeric columns where NaN means "0" — the related feature just doesn't exist
ZERO_FILL_NUMERIC = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath", "Garage Cars", "Garage Area",
]

# --- 4. Quality mapping: ordinal strings (Gd/Ex, Y/N/P) -> integers ---
QUALITY_MAPPINGS = {
    "Exter Qual":     {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Exter Cond":     {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Bsmt Qual":      {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Bsmt Cond":      {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Bsmt Exposure":  {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFin Type 1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFin Type 2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Heating QC":     {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Kitchen Qual":   {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Fireplace Qu":   {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Garage Finish":  {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Garage Qual":    {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Garage Cond":    {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Pool QC":        {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Paved Drive":    {"N": 0, "P": 1, "Y": 2},
    "Central Air":    {"N": 0, "Y": 1},
    "Functional":     {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 2, "Mod": 3, "Min2": 4, "Min1": 4, "Typ": 5},
}

# --- 1. Category collapse (rare merges), applied before OneHotEncoder ---
CATEGORY_COLLAPSE = {
    "MS Zoning":      {"C (all)": "Other", "A (agr)": "Other", "I (all)": "Other"},
    "MS SubClass":    {"40": "Other", "45": "Other", "150": "Other", "180": "Other"},
    "Lot Shape":      {"IR2": "IR1", "IR3": "IR1"},
    "Lot Config":     {"FR3": "FR2"},
    "Condition 1":    {"RRNn": "Other", "RRNe": "Other"},
    "Condition 2":    {v: "Other" for v in ["Feedr", "Artery", "PosA", "PosN", "RRNn", "RRAe", "RRAn"]},
    "Heating":        {"Grav": "Other", "Wall": "Other", "OthW": "Other", "Floor": "Other"},
    "Electrical":     {"FuseP": "Other", "Mix": "Other"},
    "Garage Type":    {"CarPort": "Other", "2Types": "Other", "Basment": "Other"},
    "Sale Type":      {v: "Other" for v in ["ConLD", "CWD", "ConLI", "ConLw", "Oth", "Con", "VWD"]},
    "Sale Condition": {"AdjLand": "Other", "Alloca": "Other"},
}

# --- 2. Skewed numeric -> PowerTransformer. TotalSF is engineered (see below)
#     but goes here too: summing skewed sqft columns stays skewed. ---
SKEWED_NUMERIC = [
    "Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2",
    "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Gr Liv Area",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "Screen Porch",
    "TotalSF",
]

ORDINAL_NUMERIC = list(QUALITY_MAPPINGS)

# --- 3. Everything else numeric -> StandardScaler. TotalBath is engineered
#     (see below) but lands here: summing small bathroom counts isn't skewed. ---
OTHER_NUMERIC = [
    "Overall Qual", "Overall Cond", "Year Built", "Year Remod/Add",
    "Low Qual Fin SF", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
    "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces",
    "Garage Yr Blt", "Garage Cars", "Garage Area", "3Ssn Porch", "Pool Area",
    "Misc Val", "Mo Sold", "Yr Sold",
    "TotalBath", "HouseAgeAtSale", "RemodAgeAtSale",
]

CATEGORICAL_NOMINAL = [
    "MS SubClass", "MS Zoning", "Street", "Alley", "Lot Shape", "Land Contour",
    "Utilities", "Lot Config", "Land Slope", "Neighborhood", "Condition 1",
    "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl",
    "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating",
    "Electrical", "Garage Type", "Fence", "Misc Feature", "Sale Type", "Sale Condition",
]


def drop_known_outliers(X, y):
    """Drops the well-documented Ames anomalies: a handful of >4000 sqft homes
    that sold far below trend (Overall Qual 10, Gr Liv Area 4600-5600, but
    SalePrice < $300k). Two OTHER large homes at similar size but ~$750k are
    genuine and are correctly kept.

    Call this on X_train/y_train only, AFTER the train/test split — never on
    test data, since dropping test rows would silently inflate reported metrics.
    """
    mask = ~((X["Gr Liv Area"] > 4000) & (y < 300000))
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Missing-value handling, quality mapping, category collapsing, and
    feature creation (basement/bath totals + house/remodel age at sale).
    Statistics are learned in `fit` on the training fold only.
    """

    def fit(self, X, y=None):
        X = X.copy()
        self.lot_frontage_by_neighborhood_ = X.groupby("Neighborhood")["Lot Frontage"].median()
        self.lot_frontage_global_median_ = X["Lot Frontage"].median()
        self.electrical_mode_ = X["Electrical"].mode(dropna=True).iloc[0]
        return self

    def transform(self, X):
        X = X.copy()

        # Drop pure identifiers
        X = X.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in X.columns])

        # MS SubClass is a numeric-looking code with no ordinal meaning -> string
        X["MS SubClass"] = X["MS SubClass"].astype(str)

        # "NA means absent" categoricals, then "NA means 0" numerics
        for col, fill_val in NONE_FILL_CATEGORICAL.items():
            X[col] = X[col].fillna(fill_val)
        X[ZERO_FILL_NUMERIC] = X[ZERO_FILL_NUMERIC].fillna(0)

        # Garage Yr Blt: no garage -> use house's own build year as a neutral fallback
        X["Garage Yr Blt"] = X["Garage Yr Blt"].fillna(X["Year Built"])

        # Lot Frontage: neighborhood-median, falling back to the global median
        # for any neighborhood unseen (or itself all-NaN) at fit time
        neighborhood_medians = X["Neighborhood"].map(self.lot_frontage_by_neighborhood_)
        X["Lot Frontage"] = X["Lot Frontage"].fillna(neighborhood_medians)
        X["Lot Frontage"] = X["Lot Frontage"].fillna(self.lot_frontage_global_median_)

        # The one stray Electrical NaN -> training mode
        X["Electrical"] = X["Electrical"].fillna(self.electrical_mode_)

        # 1. Category collapse
        for col, mapping in CATEGORY_COLLAPSE.items():
            X[col] = X[col].replace(mapping)

        # 4. Quality mapping
        for col, mapping in QUALITY_MAPPINGS.items():
            X[col] = X[col].map(mapping)

        # 5. Feature creation — basement/bath totals, temporal age-at-sale
        X["TotalBath"] = (
            X["Full Bath"] + 0.5 * X["Half Bath"]
            + X["Bsmt Full Bath"] + 0.5 * X["Bsmt Half Bath"]
        )
        X["TotalSF"] = X["Total Bsmt SF"] + X["1st Flr SF"] + X["2nd Flr SF"]
        X["HouseAgeAtSale"] = (X["Yr Sold"] - X["Year Built"]).clip(lower=0)
        X["RemodAgeAtSale"] = (X["Yr Sold"] - X["Year Remod/Add"]).clip(lower=0)

        return X


def build_preprocessing_pipeline(scale_numeric=True):
    """ColumnTransformer: PowerTransformer for skewed numeric, StandardScaler
    for the rest of numeric (both optionally), OneHotEncoder for categorical.
    """
    numeric_pipe = [("scale", StandardScaler())] if scale_numeric else []

    skewed_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
    ])
    plain_numeric_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        *numeric_pipe,
    ])
    categorical_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_transform = ColumnTransformer([
        ("skewed", skewed_transformer, SKEWED_NUMERIC),
        ("numeric", plain_numeric_transformer, ORDINAL_NUMERIC + OTHER_NUMERIC),
        ("categorical", categorical_transformer, CATEGORICAL_NOMINAL),
    ])

    return Pipeline([
        ("feature_engineer", AmesFeatureEngineer()),
        ("column_transform", column_transform),
    ])


def build_model_pipeline(model, scale_numeric=True, use_log_target=True):
    """Full X -> prediction pipeline. use_log_target=True trains on
    log1p(SalePrice) but reports predict()/CV metrics on the original scale.
    """
    preprocessing = build_preprocessing_pipeline(scale_numeric=scale_numeric)
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])
    if use_log_target:
        return TransformedTargetRegressor(
            regressor=full_pipeline, func=np.log1p, inverse_func=np.expm1
        )
    return full_pipeline