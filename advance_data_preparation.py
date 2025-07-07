import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Drop ID column
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

# ========== Add missingness indicators ==========

missing_cols = [
    'Time_spent_Alone', 'Social_event_attendance',
    'Going_outside', 'Friends_circle_size', 'Post_frequency'
]

for col in missing_cols:
    train_df[f"{col}_was_missing"] = train_df[col].isna().astype(int)
    test_df[f"{col}_was_missing"] = test_df[col].isna().astype(int)

# ========== Numeric Imputation with KNN ==========

numeric_cols = [
    'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
    'Friends_circle_size', 'Post_frequency'
]

train_numeric = train_df[numeric_cols]
test_numeric = test_df[numeric_cols]
combined_numeric = pd.concat([train_numeric, test_numeric], axis=0)

scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(combined_numeric)

imputer = KNNImputer(n_neighbors=5)
imputed_numeric = imputer.fit_transform(scaled_numeric)

# Replace in original DataFrames
imputed_df = pd.DataFrame(imputed_numeric, columns=numeric_cols)
train_df[numeric_cols] = imputed_df.iloc[:len(train_df)].values
test_df[numeric_cols] = imputed_df.iloc[len(train_df):].values

# ========== Outlier Clipping ==========
Q1 = train_df['Time_spent_Alone'].quantile(0.25)
Q3 = train_df['Time_spent_Alone'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train_df['Time_spent_Alone'] = train_df['Time_spent_Alone'].clip(lower=lower_bound, upper=upper_bound)
test_df['Time_spent_Alone'] = test_df['Time_spent_Alone'].clip(lower=lower_bound, upper=upper_bound)

# ========== Categorical Imputation ==========
for col in ['Stage_fear', 'Drained_after_socializing']:
    mode_val = train_df[col].mode()[0]
    train_df[col] = train_df[col].fillna(mode_val)
    test_df[col] = test_df[col].fillna(mode_val)

# ========== Feature Engineering ==========

def add_engineered_features(df):
    df['social_ratio'] = df['Friends_circle_size'] / (df['Post_frequency'] + 1)
    df['introversion_score'] = df['Time_spent_Alone'] + df['Stage_fear'].map({'Yes': 3, 'No': 0})
    df['social_fatigue'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0}) * df['Going_outside']
    return df

train_df = add_engineered_features(train_df)
test_df = add_engineered_features(test_df)

# ========== Save Cleaned Data ==========
train_df.to_csv("cleaned_train.csv", index=False)
test_df.to_csv("cleaned_test.csv", index=False)
