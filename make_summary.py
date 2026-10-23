import pandas as pd
from run_five_seeds import summarize

df = pd.read_csv("five_seed_cpp_clean.csv")
summary = summarize(df)
summary.to_csv("five_seed_cpp_clean_summary.csv", index=False)
print("\n[✓] Successfully generated five_seed_cpp_clean_summary.csv!")
