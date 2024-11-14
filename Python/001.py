import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
file_path = 'your_file.csv'  # replace with your file path
df = pd.read_csv(file_path)

# Handle missing values
# Impute missing numerical values with the median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.fillna(x.median()))

# Impute missing categorical values with the mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Remove duplicates
df = df.drop_duplicates()

# Normalize numerical columns between 0 and 1
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Convert categorical columns to one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save the processed data to a new CSV file
df.to_csv('processed_data.csv', index=False)
