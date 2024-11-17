import pandas as pd

df = pd.read_pickle("output.pkl")
split_index = len(df) // 2
df_part1 = df.iloc[:split_index]  # First half
df_part2 = df.iloc[split_index:]  # Second half
df_part1.to_pickle("part1.pkl")
df_part2.to_pickle("part2.pkl")
