import pandas as pd



df = pd.read_csv("data_clean_complete_4_loop.csv", header=0, index_col=0)

print("Data preview:")
print(df.head())

print("\nDescriptive statistics:")
print(df.describe())

print("\nDataset info:")
print(df.info())

# Count missing values per column
missing_values = df.isna().sum()

# Show only columns that have missing values
print("Missing values per column:")
print(missing_values[missing_values > 0])
