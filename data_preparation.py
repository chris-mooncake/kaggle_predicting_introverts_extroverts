import pandas as pd

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Drop ID column
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

# ========== Numerical variables ==========

# Time_spent_Alone: fill NAs and cap outliers
train_df['Time_spent_Alone'] = train_df['Time_spent_Alone'].fillna(train_df['Time_spent_Alone'].median())
test_df['Time_spent_Alone'] = test_df['Time_spent_Alone'].fillna(train_df['Time_spent_Alone'].median())

Q1 = train_df['Time_spent_Alone'].quantile(0.25)
Q3 = train_df['Time_spent_Alone'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train_df['Time_spent_Alone'] = train_df['Time_spent_Alone'].clip(lower=lower_bound, upper=upper_bound)
test_df['Time_spent_Alone'] = test_df['Time_spent_Alone'].clip(lower=lower_bound, upper=upper_bound)

# Friends_circle_size
train_df['Friends_circle_size'] = train_df['Friends_circle_size'].fillna(train_df['Friends_circle_size'].median())
test_df['Friends_circle_size'] = test_df['Friends_circle_size'].fillna(train_df['Friends_circle_size'].median())

# Post_frequency
train_df['Post_frequency'] = train_df['Post_frequency'].fillna(train_df['Post_frequency'].median())
test_df['Post_frequency'] = test_df['Post_frequency'].fillna(train_df['Post_frequency'].median())

# Social_event_attendance
train_df['Social_event_attendance'] = train_df['Social_event_attendance'].fillna(train_df['Social_event_attendance'].mean())
test_df['Social_event_attendance'] = test_df['Social_event_attendance'].fillna(train_df['Social_event_attendance'].mean())

# Going_outside
train_df['Going_outside'] = train_df['Going_outside'].fillna(train_df['Going_outside'].mean())
test_df['Going_outside'] = test_df['Going_outside'].fillna(train_df['Going_outside'].mean())

# ========== Categorical variables ==========

# Fill missing categorical values
train_df['Stage_fear'] = train_df['Stage_fear'].fillna(train_df['Stage_fear'].mode()[0])
test_df['Stage_fear'] = test_df['Stage_fear'].fillna(train_df['Stage_fear'].mode()[0])

train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].fillna(train_df['Drained_after_socializing'].mode()[0])
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].fillna(train_df['Drained_after_socializing'].mode()[0])

# ========== Optional: Save cleaned data ==========
train_df.to_csv("cleaned_train.csv", index=False)
test_df.to_csv("cleaned_test.csv", index=False)
