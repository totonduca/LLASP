import pandas as pd

# Paths to the two CSV files
csv_file_1 = 'data/val_invariance.csv'
csv_file_2 = 'data/val_basecomplex.csv'

# Load the CSV files into DataFrames
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

# Merge the two DataFrames
# Assuming you want to concatenate them row-wise
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_csv_file = 'data/val_basecomplex_2.csv'
merged_df.to_csv(merged_csv_file, index=False)

print(f"Merged CSV saved to {merged_csv_file}")

# Print the lengths of the individual DataFrames
print(f"Length of first CSV file: {len(df1)}")
print(f"Length of second CSV file: {len(df2)}")

# Print the length of the merged DataFrame
print(f"Length of merged CSV file: {len(merged_df)}")