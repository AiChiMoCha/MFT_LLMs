import pandas as pd

in_path = "sae_feature_table_with_interpretations_gpt-5.1_qwen.csv"
out_path = "sae_feature_table_with_interpretations_gpt-5.1_qwen.filtered.csv"

# Read CSV
df = pd.read_csv(in_path)

# Normalize "mft_alignment" so that various "none-like" values become missing
# (handles: None/NaN, "none", "None", "", "null", "nan", etc.)
df["mft_alignment"] = (
    df["mft_alignment"]
      .astype(str)                 # safe even if mixed types
      .str.strip()                 # remove surrounding whitespace
)

none_like = {"none", "null", "nan", ""}

mask = df["mft_alignment"].notna() & ~df["mft_alignment"].str.lower().isin(none_like)

# Keep only rows where mft_alignment is not none-like
df_filtered = df[mask].copy()

# Save
df_filtered.to_csv(out_path, index=False)

print(f"Original rows: {len(df):,}")
print(f"Filtered rows: {len(df_filtered):,}")
print(f"Wrote: {out_path}")
