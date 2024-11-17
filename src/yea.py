import pandas as pd

# Load the original DataFrame from the pickle file
df = pd.read_pickle("output.pkl")

# Split the DataFrame into two parts (row-wise)
split_index = len(df) // 2
df_part1 = df.iloc[:split_index]  # First half
df_part2 = df.iloc[split_index:]  # Second half

# Save the two parts into separate pickle files
df_part1.to_pickle("part1.pkl")
df_part2.to_pickle("part2.pkl")
