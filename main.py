import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
)
import warnings

from preprocess import del_cols, get_date_diff

warnings.filterwarnings('ignore')

raw_train = pd.read_csv("dataset/train.csv")
raw_test = pd.read_csv("dataset/test.csv")
raw_sub = pd.read_csv("dataset/sample_submission.csv")

# make all cost positive and remove outliers
df_train = raw_train.copy()
df_train = raw_train.loc[
    ~(
        (raw_train["Cost"] > 6000000)
        | (raw_train["Price Of Sculpture"] > 300000)
        | (raw_train["Weight"] > 100000000)
    )
]
df_train["Cost"] = df_train["Cost"].apply(abs)


X = df_train.drop("Cost", axis=1)
y = df_train["Cost"]

num_features = [
    "Artist Reputation",
    "Height",
    "Width",
    "Weight",
    "Price Of Sculpture",
    "Base Shipping Price",
]
cat_features = ["Material", "Transport", "Remote Location"]
zero_one_features = [
    "International",
    "Express Shipment",
    "Installation Included",
    "Customer Information",
    "Fragile",
]

# transformers
numeric_transformer = Pipeline(
    steps=[
        ("num_imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
        ("scalar", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    [
        ("cat_imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("oh_encoder", OneHotEncoder()),
    ]
)

col_transformer = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
        ("zo", OrdinalEncoder(), zero_one_features),
    ],
    remainder="passthrough",
)

full_pipeline = Pipeline(
    [
        ("del_col", FunctionTransformer(del_cols)),
        ("get_date_diff", FunctionTransformer(get_date_diff)),
        ("pre", col_transformer),
    ]
)

regressor = GradientBoostingRegressor(
    n_estimators=261, max_depth=7, learning_rate=0.065
)
model = TransformedTargetRegressor(
    regressor=regressor, transformer=QuantileTransformer(output_distribution="normal")
)

classifier = Pipeline(
    steps=[
        ("preprocess", full_pipeline),
        ("model", model),
    ]
)
print('Cross validating the model.....')
scores = cross_val_score(
    classifier, X, y, cv=5, scoring="neg_mean_squared_log_error", n_jobs=-1
)
print(f'Scores: {scores}')
print(f"Mean Score: {(1 + scores.mean()) * 100:.4f}")

print(f'Fitting the model on entire training data.....')
classifier.fit(X, y)
print(f'Fitting done!! Getting predictions of test data.....')
y_preds = classifier.predict(raw_test)

sub = pd.DataFrame({"Customer Id": raw_test["Customer Id"].values, "Cost": y_preds})
sub.head()

sub.to_csv("my_submission.csv", index=False)
print(f'Predictions saved successfully!')
