'''
This file plots the average cosine similarity of top SAE features with MFT concept vectors,
including a PER-LAYER random baseline for comparison.

The baseline varies by layer because each layer's SAE decoder has different properties.
This properly controls for layer-specific effects.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Define Custom Colors ---
COLORS = {
    'care': '#E91E63',       # Pink
    'fairness': '#2196F3',   # Blue
    'loyalty': '#4CAF50',    # Green
    'authority': '#FF9800',  # Orange
    'sanctity': '#9C27B0'    # Purple
}

# 2. Load Data
df = pd.read_csv("mft_sae_feature_candidates.csv")
df_baseline = pd.read_csv("mft_sae_random_baseline.csv")

df['Layer'] = df['Layer'] - 1
df_baseline['Layer'] = df_baseline['Layer'] - 1

# Normalize foundation names
df['Foundation'] = df['Foundation'].str.lower()
df = df[df['Foundation'].isin(COLORS.keys())]

print(f"Foundations found: {df['Foundation'].unique()}")
print(f"Layers in baseline: {sorted(df_baseline['Layer'].unique())}")

# 3. Sort carefully
df_sorted = df.sort_values(
    by=["Foundation", "Layer", "Cosine_Similarity"], 
    ascending=[True, True, False]
)

# 4. Filter and Plot for different top_nums
top_nums = [3, 5, 10]

for top_num in top_nums:
    # Filter: Top Features PER Foundation PER Layer
    df_top = df_sorted.groupby(["Foundation", "Layer"]).head(top_num)
    
    # Aggregate: Calculate Average for the Top foundations
    df_avg = df_top.groupby(["Foundation", "Layer"])["Cosine_Similarity"].mean().reset_index()
    
    # Compute baseline average for the same top_num (PER LAYER)
    df_baseline_filtered = df_baseline[df_baseline["Rank_in_Layer"] <= top_num]
    df_baseline_avg = df_baseline_filtered.groupby("Layer").agg({
        "Cosine_Similarity": "mean",
        "Cosine_Similarity_Std": "mean"
    }).reset_index()
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot foundation lines
    sns.lineplot(
        data=df_avg,
        x="Layer",
        y="Cosine_Similarity",
        hue="Foundation",
        marker="o",
        linewidth=2.5,
        palette=COLORS,
        ax=ax
    )
    
    # Plot per-layer random baseline as gray dashed line
    ax.plot(
        df_baseline_avg["Layer"], 
        df_baseline_avg["Cosine_Similarity"],
        color='gray',
        linestyle='--',
        linewidth=2.5,
        marker='s',
        markersize=7,
        label='Random Baseline',
        zorder=1
    )
    
    # Add shaded region for baseline ± 1 std
    ax.fill_between(
        df_baseline_avg["Layer"],
        df_baseline_avg["Cosine_Similarity"] - df_baseline_avg["Cosine_Similarity_Std"],
        df_baseline_avg["Cosine_Similarity"] + df_baseline_avg["Cosine_Similarity_Std"],
        color='gray',
        alpha=0.2,
        zorder=0
    )
    
    ax.set_title(f"Average Alignment of Top-{top_num} Features Across Layers", fontsize=14)
    ax.set_ylabel("Avg Cosine Similarity", fontsize=14)
    ax.set_xlabel("Layer Index", fontsize=14)
    
        # Update legend
    ax.legend(loc='upper left', fontsize=14)

    # IMPORTANT: set ticks first
    ax.set_xticks(sorted(df_avg["Layer"].unique()))

    # === ADD THIS BLOCK ===
    ax.grid(
        which="major",
        axis="both",      # vertical (x) + horizontal (y) lines
        linestyle="--",
        linewidth=0.8,
        alpha=0.6
    )
    # === END ADDITION ===

    plt.tight_layout()
    plt.savefig(
        f"avg_cos_sim_top_{top_num}_with_baseline_llama.pdf",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Top-{top_num} Summary")
    print('='*60)
    
    # Per-layer comparison
    print(f"\nPer-Layer Comparison:")
    print(f"{'Layer':<8} {'Foundation Avg':<16} {'Baseline':<12} {'Lift':<8}")
    print("-" * 50)
    
    for layer in sorted(df_avg["Layer"].unique()):
        foundation_avg = df_avg[df_avg["Layer"] == layer]["Cosine_Similarity"].mean()
        baseline_val = df_baseline_avg[df_baseline_avg["Layer"] == layer]["Cosine_Similarity"].values[0]
        lift = foundation_avg / baseline_val
        print(f"{layer:<8} {foundation_avg:<16.4f} {baseline_val:<12.4f} {lift:<8.2f}x")
    
    # Overall statistics
    overall_foundation_avg = df_avg["Cosine_Similarity"].mean()
    overall_baseline_avg = df_baseline_avg["Cosine_Similarity"].mean()
    print(f"\nOverall:")
    print(f"  Foundation average: {overall_foundation_avg:.4f}")
    print(f"  Baseline average: {overall_baseline_avg:.4f}")
    print(f"  Average lift: {overall_foundation_avg / overall_baseline_avg:.2f}x")
    print(f"  Peak foundation: {df_avg['Cosine_Similarity'].max():.4f}")
    print(f"  Peak lift: {df_avg['Cosine_Similarity'].max() / overall_baseline_avg:.2f}x")
