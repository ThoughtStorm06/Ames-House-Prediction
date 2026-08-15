# Ames Housing — SalePrice Prediction Pipeline

## Project structure

```
.
├── Ames_Housing_Final.ipynb     # main analysis & modeling notebook
├── ames_features.py             # reusable, target-free feature engineering (imported by the notebook)
├── model.pkl                    # serialized final pipeline (created when the notebook is run)
└── sources/
    └── AmesHousing.csv          # raw data (not included — see Data below)
```

## 1. Problem

Predict `SalePrice` — a continuous dollar value — for a residential property in Ames, Iowa, from its physical, quality, and location attributes (lot size, room counts, quality ratings, garage/basement details, neighborhood, sale terms, and so on). This is a supervised regression problem, evaluated in original dollars (MAE, RMSE) and in R², so the error numbers stay interpretable as "how far off, in dollars" rather than an abstract loss value.

## 2. Data

The [Ames Housing dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset) (Dean De Cock's alternative to the classic Boston Housing set): 2,930 property sales in Ames, Iowa between 2006–2010, with roughly 80 raw columns spanning zoning, lot configuration, quality/condition ratings, room counts, basement and garage detail, porch/deck area, and sale conditions.

An earlier exploratory pass on this data mixed pandas with manual analysis in Excel. That Excel work — the pivot tables and charts that originally justified certain category groupings — wasn't preserved alongside the code. Section 5 below (`Reconstructing the lost analysis`) explains how this pipeline handles that gap honestly rather than just re-asserting the old groupings on faith.

The notebook expects the raw CSV at `sources/AmesHousing.csv` (or `AmesHousing.csv` in the same folder).

## 3. Requirements

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `joblib`. Random seeds are fixed throughout (`RANDOM_STATE = 42`) so re-running the notebook reproduces the same split, folds, and results.

## 4. Workflow, and why each step happens in this order

### Step 1 — Split before touching anything else

The raw data is split 80/20 into a development set and a held-out test set **immediately after loading**, before a single chart is drawn or a single encoding decision is made. `Order` and `PID` are excluded — they're record/parcel identifiers, not house attributes.

*Why first:* any decision informed by looking at the full dataset — including the rows that will later be "held out" — risks quietly tuning the pipeline to data it's supposed to be tested against. Splitting first and treating the test 20% as untouched until the very end is what makes the final test score a trustworthy estimate of real-world performance, rather than a number the pipeline was indirectly shaped around.

### Step 2 — Initial audit

Row/column counts, numeric vs. categorical column counts, duplicate rows, and missing targets are checked before any modeling.

*Why:* cheap structural checks catch problems (duplicate records, an unexpectedly missing target) before they propagate silently through ~80 columns of feature engineering.

### Step 3 — Exploratory data analysis (development data only)

Target distribution, the strongest absolute correlations with `SalePrice`, quality/living-area scatterplots, a neighborhood price comparison, and a missingness ranking — all computed on the 80% development split, never the test set.

*Why:* confirms the two variables everyone would expect to matter (`Overall Qual`, `Gr Liv Area`) actually do, and shows that missing values cluster in a way that suggests "this amenity doesn't exist" (e.g., no pool) rather than a data-entry error — which is what justifies encoding those as 0 later instead of dropping rows or imputing a mean.

### Step 4 — Reconstructing the lost pivot-table analysis

For every category the original pass consolidated (e.g., collapsing rare railroad-adjacency labels into `RR`, rare zoning codes into `Other`), this step rebuilds a count / median-price / mean-price table per category, directly in pandas, using development data only.

*Why:* the original consolidation decisions were real judgment calls, but the Excel analysis behind them is gone. Rather than silently re-apply the same groupings without justification, this step re-derives a transparent, reproducible version of that reasoning — so the groupings are validated against actual price patterns in the current pipeline, not just inherited on faith.

### Step 5 — Feature engineering (`ames_features.py`)

All feature logic lives in `AmesFeatureEngineer`, an sklearn-compatible transformer (`fit`/`transform`) so it can sit inside a `Pipeline` and get refit correctly inside every cross-validation fold — it never sees data it shouldn't.

| What it does | Why |
|---|---|
| Drops `Order`, `PID` | Identifiers, not house attributes |
| Casts `MS SubClass` to categorical | It's a dwelling-type code (e.g., `60` = "2-Story 1946 & Newer"), not a magnitude — treating it as numeric would imply a code of 90 is "more" than a code of 20 |
| `CATEGORY_COLLAPSE` — consolidates rare/related labels | Same groupings validated in Step 4; reduces sparse, rarely-seen categories that a model can't learn much from individually |
| `QUALITY_MAPPINGS` — converts ordered labels (`Po`/`Fa`/`TA`/`Gd`/`Ex`, etc.) to a 0–5 numeric scale | Lets the model use the *order* (Excellent > Good > Average) instead of treating each grade as an unrelated category, which is what one-hot encoding would otherwise do |
| `TotalSF`, `TotalBath`, `TotalPorchSF` | Aggregates related columns (basement + 1st floor + 2nd floor square footage; half-baths counted as 0.5) into single, less noisy signals |
| `HouseAgeAtSale`, `RemodAgeAtSale` | Converts absolute years (`Year Built = 1920`) into age relative to the sale — it's age at the time of sale that drives price, not the raw calendar year |
| `HasGarage`, `HasBasement`, `HasFireplace` | Clean binary signal for "has this amenity at all," which especially helps linear models that would otherwise see mostly zeros in the continuous version |

### Step 6 — Preprocessing

A `ColumnTransformer`, selecting columns by dtype *after* feature engineering: numeric columns get median imputation + `StandardScaler`; remaining categorical columns get constant `"Missing"` imputation + one-hot encoding (`handle_unknown="ignore"`, so a category seen only at prediction time becomes all-zeros instead of raising an error).

The target itself is modeled as `log1p(SalePrice)` via `TransformedTargetRegressor`, then converted back to dollars automatically on every `.predict()` call — `SalePrice` is right-skewed, and modeling its log keeps errors closer to constant-variance across the price range, while every metric reported still reads in real dollars.

### Step 7 — Baseline and candidate models, tested across multiple validation sets

Five candidates are compared: `DummyRegressor(strategy="median")` as a floor, `Ridge`, `Elastic Net`, `Extra Trees`, and `Random Forest`.

**How "multiple validation sets" works here:** every candidate is scored with the same 5-fold `KFold` split (shuffled, `random_state=42`) — meaning each model is fit and validated **5 separate times**, once per fold, on 5 different train/validation partitions of the development data, and the reported CV MAE / RMSE / R² is the *average* across those 5 runs (with the spread reported too, as `CV R² SD`).

*Why 5 separate validation sets instead of 1:* a single train/validation split can make a mediocre model look good — or a good one look bad — purely by chance, depending on which rows happened to land in validation. Averaging across 5 different partitions gives a much more stable read on how a model actually generalizes. `return_train_score=True` also records each fold's training score, so a model that fits its training folds far better than it validates (a wide Train R² − CV R² gap) gets flagged as likely overfitting, even if its raw CV score looks competitive.

```python
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"mae": "neg_mean_absolute_error", "rmse": make_scorer(rmse, greater_is_better=False), "r2": "r2"}
# every model below is fit and scored once per fold, on the same 5 folds:
scores = cross_validate(make_estimator(model), X_train, y_train, cv=CV, scoring=SCORING, return_train_score=True)
```

### Step 8 — Hyperparameter tuning via cross-validated random search

`RandomizedSearchCV` samples parameter combinations for each non-baseline model family (`Ridge`, `Elastic Net`, `Extra Trees`, and `Random Forest`), and scores **every sampled combination using the identical 5-fold CV from Step 7** — same folds, same scorers — rather than a single train/validation split.

*Why CV instead of one split, here too:* tuning against a single split risks picking hyperparameters that happen to fit that one partition well, not ones that generalize. Scoring each candidate across all 5 folds and averaging keeps the search honest.

```python
SEARCH_SPACES = {
    "Ridge": (Ridge(), {
        "regressor__model__alpha": loguniform(1e-1, 1e3),
    }),
    "Elastic Net": (ElasticNet(max_iter=20_000, random_state=RANDOM_STATE), {
        "regressor__model__alpha": loguniform(1e-4, 1e1),
        "regressor__model__l1_ratio": uniform(0.05, 0.90),
    }),
    "Extra Trees": (ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1), {
        "regressor__model__n_estimators": randint(150, 351),
        "regressor__model__max_depth": [None, 10, 16, 24, 32],
        "regressor__model__min_samples_split": randint(2, 16),
        "regressor__model__min_samples_leaf": randint(1, 7),
        "regressor__model__max_features": ["sqrt", .50, .75, 1.0],
    }),
    "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), {
        "regressor__model__n_estimators": randint(150, 351),
        "regressor__model__max_depth": [None, 10, 16, 24, 32],
        "regressor__model__min_samples_split": randint(2, 16),
        "regressor__model__min_samples_leaf": randint(1, 7),
        "regressor__model__max_features": ["sqrt", .50, .75, 1.0],
    }),
}
```

### Step 9 — Model comparison and selection

All candidate models (both default and tuned) are combined into a final comparison table and sorted by CV RMSE. The lowest CV RMSE wins, subject to a reasonable train/CV gap — **not** the training score, and **not** the test score, since the test set still hasn't been touched.

### Step 10 — Held-out test evaluation (once)

The selected model, refit on the full development set, is scored on the 20% held out since Step 1 — the only point in the whole notebook where those rows are used. Actual-vs-predicted and residual plots check for systematic bias; the largest-error rows are pulled out for manual review, explicitly *not* as a cue to retune anything.

### Step 11 — Feature importance

For the tree-based winner, impurity-based importances are listed with an explicit caveat: importance is descriptive/associative, not causal, and it's biased toward high-cardinality one-hot columns (like `Neighborhood`) — worth pairing with domain knowledge, not reading as "why" on its own.

### Step 12 — Serialization

The final pipeline — feature engineering, preprocessing, and model, as one object — is saved to `model.pkl` via `joblib`, then immediately reloaded and re-predicted on a sample row to confirm the saved artifact reproduces the in-memory pipeline exactly. This catches silent serialization bugs before they reach anyone downstream.

```python
import joblib
model = joblib.load("model.pkl")
model.predict(new_raw_dataframe)   # raw Ames-schema columns in; no manual preprocessing needed
```
`ames_features.py` must be importable (on the Python path) wherever `model.pkl` is loaded, since the pickle references the `AmesFeatureEngineer` class defined there.

## 5. Results

**5-fold cross-validation, development data only:**

| Model | CV MAE | CV RMSE | CV R² | CV R² SD | Train R² | Note |
|---|---:|---:|---:|---:|---:|---|
| Median baseline | $54,123 | $79,308 | −0.060 | 0.015 | −0.059 | Floor — ignores every feature |
| Elastic Net | $16,251 | $46,790 | 0.227 | 1.382 | 0.896 | Default hyperparameters |
| Ridge | $16,121 | $45,916 | 0.267 | 1.302 | 0.911 | Default hyperparameters |
| Tuned Ridge | $15,616 | $43,424 | 0.341 | 1.174 | 0.953 | Search improved CV RMSE over default Ridge |
| Tuned Elastic Net | $15,079 | $42,480 | 0.357 | 1.154 | 0.950 | Search improved CV RMSE over default Elastic Net |
| Tuned Random Forest | $15,737 | $26,422 | 0.878 | 0.050 | 0.977 | Tuning slightly reduced train-CV gap |
| Random Forest | $15,768 | $26,310 | 0.879 | 0.051 | 0.984 | |
| Extra Trees | $15,001 | $25,480 | 0.886 | 0.049 | 1.000 | |
| **Tuned Extra Trees** | **$14,989** | **$25,209** | **0.888** | **0.049** | **0.996** | **Selected** — lowest CV RMSE |

**Final, one-time test evaluation (Tuned Extra Trees):** MAE $14,810 · RMSE $24,252 · R² 0.927.

## 6. Per-model insight

- **Median baseline** exists purely as a floor: every real model needs to clear its ~$79k RMSE by a wide margin, or the features aren't adding information.
- **Ridge / Elastic Net** improved with hyperparameter tuning (CV R² rose from ~0.23–0.27 to ~0.34–0.36, CV RMSE dropped to ~$42–43k), but remain substantially behind tree ensembles. Housing prices in Ames exhibit threshold effects and non-linear interactions (e.g., quality ratings, specific neighborhood premiums) that linear models struggle to capture without manual interaction terms.
- **Random Forest / Extra Trees** both achieve CV R² around 0.88 with CV RMSE around $25k–$26k, demonstrating strong predictive performance on non-linear feature interactions.
- **Tuned Extra Trees** achieved the lowest CV RMSE ($25,209) and lowest CV MAE ($14,989) with a small reduction in the training overfit gap (Train R² 0.996 vs 1.000 on default Extra Trees), making it the selected final estimator.
- **Generalization:** The one-time held-out test evaluation confirmed the CV findings with Test RMSE $24,252, Test MAE $14,810, and Test R² 0.927.

## 7. Known limitations — open, not resolved

- **Linear model interaction terms:** While regularized linear models were tuned over `alpha` and `l1_ratio`, they lack explicit polynomial and interaction terms needed to model non-linear price curves and multi-way feature interactions.
- **Hyperparameter search budget:** `RandomizedSearchCV` sampled 8 iterations per model family. Expanding search budgets or employing Bayesian optimization (e.g., Optuna) could further optimize tree ensemble hyperparameters.
- **Impurity-based feature importance:** The reported feature importance uses Gini/variance reduction, which inherently skews toward high-cardinality features (like one-hot encoded `Neighborhood`). Permutation importance or SHAP values on a validation split would provide a more robust attribution.
