import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from AmesFeatures import (
    AmesFeatureEngineer,
    build_preprocessing_pipeline,
    drop_known_outliers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "AmesHousing.csv"


class AmesFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(DATA_PATH)
        cls.X = cls.raw.drop(columns=["SalePrice"])
        cls.y = cls.raw["SalePrice"]

    def test_outlier_removal_keeps_normal_rows_and_removes_anomalies(self):
        X = pd.DataFrame({"Gr Liv Area": [3999, 4001, 5000]})
        y = pd.Series([250000, 250000, 299999])

        filtered_X, filtered_y = drop_known_outliers(X, y)

        self.assertEqual(filtered_X["Gr Liv Area"].tolist(), [3999])
        self.assertEqual(filtered_y.tolist(), [250000])

    def test_feature_engineer_creates_expected_features(self):
        engineer = AmesFeatureEngineer().fit(self.X.head(1))
        engineered = engineer.transform(self.X.head(1))
        row = engineered.iloc[0]

        expected_total_sf = row["Total Bsmt SF"] + row["1st Flr SF"] + row["2nd Flr SF"]
        expected_total_bath = (
            row["Full Bath"]
            + 0.5 * row["Half Bath"]
            + row["Bsmt Full Bath"]
            + 0.5 * row["Bsmt Half Bath"]
        )

        self.assertEqual(row["TotalSF"], expected_total_sf)
        self.assertEqual(row["TotalBath"], expected_total_bath)
        self.assertNotIn("Order", engineered.columns)
        self.assertNotIn("PID", engineered.columns)

    def test_preprocessing_handles_missing_values_and_unseen_categories(self):
        train = self.X.iloc[:100].copy()
        test = self.X.iloc[100:110].copy()
        test["Neighborhood"] = "UnseenNeighborhood"
        test.loc[test.index[0], "Lot Frontage"] = np.nan

        pipeline = build_preprocessing_pipeline()
        pipeline.fit(train, self.y.iloc[:100])
        transformed = pipeline.transform(test)

        self.assertEqual(transformed.shape[0], len(test))
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
