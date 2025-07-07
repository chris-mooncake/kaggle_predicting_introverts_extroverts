import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Train df head:")
print(train_df.head())
print("\nTest df head:")
print(test_df.head())

print(train_df.isna().sum())
print(test_df.isna().sum())

# Checking numerical variables for distribution

numerical_cols = [
    'Time_spent_Alone', 
    'Social_event_attendance', 
    'Going_outside', 
    'Friends_circle_size', 
    'Post_frequency'
]

# train_df[numerical_cols].hist(bins=20, figsize=(12, 8))
# plt.tight_layout()
# plt.show()

# train_df[numerical_cols].plot(kind='box', subplots=True, layout=(2, 3), figsize=(12, 6), sharex=False, sharey=False)
# plt.tight_layout()
# plt.show()

print(train_df[numerical_cols].describe())

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)]


for i in numerical_cols:
    outliers = detect_outliers_iqr(train_df, i)
    print(f"{i}: {outliers.shape}")