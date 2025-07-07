import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Load cleaned data
train_df = pd.read_csv("data/cleaned_train.csv")
test_df = pd.read_csv("data/cleaned_test.csv")

# Separate features and target
X = train_df.drop(columns=['Personality'])
y = train_df['Personality']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Preprocessing steps
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# Define model pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(use_label_encoder=False, random_state=42))
])

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Validate
y_pred = pipeline.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Final model on full training data
pipeline.fit(X, y)

# Predict on test set
test_predictions = pipeline.predict(test_df)

# Save submission file
submission = pd.DataFrame({
    "id": range(18524, 18524 + len(test_predictions)),
    "Personality": test_predictions
})

# Map back to original labels if needed
submission['Personality'] = submission['Personality'].map({0: 'Introvert', 1: 'Extrovert'})

submission.to_csv("submission.csv", index=False)
