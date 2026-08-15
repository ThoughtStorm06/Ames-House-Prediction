"""Deterministic, target-free feature engineering for the Ames Housing dataset.

Kept in its own module (rather than defined inline in the notebook) so that
``model.pkl`` can be unpickled in a fresh Python process: joblib stores a
class by its module path + name, and a class defined inside a notebook lives
in ``__main__``, which does not exist outside that kernel. Any process that
loads model.pkl just needs this file importable alongside it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Related/rare labels collapsed into one level. Fixed, target-free rules
# reconstructed from the manual Excel groupings in the original analysis.
CATEGORY_COLLAPSE = {
    "Condition 1": {"RRNe": "RR", "RRAe": "RR", "RRAn": "RR", "RRNn": "RR", "PosN": "Pos", "PosA": "Pos"},
    "MS Zoning": {"RH": "Other", "C (all)": "Other", "I (all)": "Other", "A (agr)": "Other"},
    "Lot Config": {"FR2": "FR"},
    "Heating": {"Grav": "Other", "Wall": "Other", "Floor": "Other", "OthW": "Other"},
    "Electrical": {"FuseP": "Other", "Mix": "Other"},
    "Functional": {"Min1": "Min", "Min2": "Min", "Maj1": "Maj", "Maj2": "Maj"},
}

# Ordered quality/condition scale shared by most "Ex/Gd/TA/Fa/Po" columns; absent -> 0.
_FIVE_LEVEL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

QUALITY_MAPPINGS = {
    "Pool QC": _FIVE_LEVEL,
    "Garage Qual": _FIVE_LEVEL,
    "Garage Cond": _FIVE_LEVEL,
    "Fireplace Qu": _FIVE_LEVEL,
    "Kitchen Qual": _FIVE_LEVEL,
    "Heating QC": _FIVE_LEVEL,
    "Bsmt Qual": _FIVE_LEVEL,
    "Bsmt Cond": _FIVE_LEVEL,
    "Exter Qual": _FIVE_LEVEL,
    "Exter Cond": _FIVE_LEVEL,
    "Bsmt Exposure": {"None": 0, "No": 0, "Mn": 1, "Av": 2, "Gd": 3},
    "BsmtFin Type 1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFin Type 2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Garage Finish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Paved Drive": {"N": 0, "P": 1, "Y": 2},
    "Central Air": {"N": 0, "Y": 1},
}

_PORCH_COLUMNS = ["Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Wood Deck SF"]


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Apply deterministic feature engineering to a raw Ames Housing DataFrame."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AmesFeatureEngineer requires a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        out = X.copy().drop(columns=["Order", "PID"], errors="ignore")

        if "MS SubClass" in out:
            out["MS SubClass"] = out["MS SubClass"].astype("string")

        for column, mapping in CATEGORY_COLLAPSE.items():
            if column in out:
                out[column] = out[column].replace(mapping)

        for column, mapping in QUALITY_MAPPINGS.items():
            if column not in out:
                continue
            mapped = out[column].fillna("None").map(mapping)
            unknown = out.loc[mapped.isna(), column].dropna().unique()
            if len(unknown):
                raise ValueError(f"Unexpected level(s) in {column}: {unknown.tolist()}")
            out[column] = mapped.astype(float)

        out["TotalSF"] = (
            out["Total Bsmt SF"].fillna(0) + out["1st Flr SF"].fillna(0) + out["2nd Flr SF"].fillna(0)
        )
        out["TotalBath"] = (
            out["Full Bath"].fillna(0)
            + 0.5 * out["Half Bath"].fillna(0)
            + out["Bsmt Full Bath"].fillna(0)
            + 0.5 * out["Bsmt Half Bath"].fillna(0)
        )
        out["TotalPorchSF"] = out[_PORCH_COLUMNS].fillna(0).sum(axis=1)
        out["HouseAgeAtSale"] = (out["Yr Sold"] - out["Year Built"]).clip(lower=0)
        out["RemodAgeAtSale"] = (out["Yr Sold"] - out["Year Remod/Add"]).clip(lower=0)
        out["HasGarage"] = (out["Garage Area"].fillna(0) > 0).astype(int)
        out["HasBasement"] = (out["Total Bsmt SF"].fillna(0) > 0).astype(int)
        out["HasFireplace"] = (out["Fireplaces"].fillna(0) > 0).astype(int)
        return out
