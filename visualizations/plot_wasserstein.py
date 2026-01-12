"""
Wasserstein Distance Comparison for All Moral Foundations
WITH DIRECTION - negative values indicate wrong direction
PDF output only
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== Configuration ====================
BASE_DIR = "/data/cyu/MFT_LLM_ARR/projection_results_P1"

RESULT_DIRS = {
    'care': f"{BASE_DIR}/care_vs_social/care_vs_nonmoral_2000_20251225_221904",
    'fairness': f"{BASE_DIR}/fairness_vs_social/fairness_vs_nonmoral_2000_20251225_222046",
    'loyalty': f"{BASE_DIR}/loyalty_vs_social/loyalty_vs_nonmoral_2000_20251225_222327",
    'authority': f"{BASE_DIR}/authority_vs_social/authority_vs_nonmoral_2000_20251225_222910",
    'sanctity': f"{BASE_DIR}/sanctity_vs_social/sanctity_vs_nonmoral_2000_20251225_223007",
}

OUTPUT_PATH = './wasserstein_all_foundations_with_direction.pdf'

# ==================== Styling ====================
COLORS = {
    'care': '#E91E63',        # Pink
    'fairness': '#2196F3',    # Blue
    'loyalty': '#4CAF50',     # Green
    'authority': '#FF9800',   # Orange
    'sanctity': '#9C27B0'     # Purple
}

MARKERS = {
    'care': 'o',
    'fairness': 's',
    'loyalty': '^',
    'authority': 'D',
    'sanctity': 'v'
}

# ==================== Functions ====================
def load_wasserstein_with_direction(stats_file: str):
    """Load Wasserstein distances WITH direction (negative if wrong direction)"""
    with open(stats_file, 'r') as f:
        data = json.load(f)
    
    wass_dict = {}
    
    for key, value in data.items():
        if key.startswith('layer_'):
            layer_idx = int(key.split('_')[1])
            wass_d = value['wass']['D']
            correct_direction = value['direction']['correct_direction']
            
            # If direction is wrong, make it negative
            if not correct_direction:
                wass_d = -wass_d
            
            wass_dict[layer_idx] = wass_d
    
    return wass_dict

# ==================== Main Script ====================
print("="*70)
print("Generating Signed Wasserstein Distance Comparison Plot")
print("="*70)

# Collect all data
all_data = {}
for foundation, result_dir in RESULT_DIRS.items():
    stats_file = Path(result_dir) / "statistics.json"
    
    if not stats_file.exists():
        print(f"⚠️ Warning: {stats_file} not found!")
        continue
    
    wass_dict = load_wasserstein_with_direction(str(stats_file))
    all_data[foundation] = wass_dict
    print(f"✅ Loaded {foundation}: {len(wass_dict)} layers")

if not all_data:
    print("❌ No data loaded!")
    exit(1)

# ==================== Plotting ====================
fig, ax = plt.subplots(figsize=(16, 9))

# Plot each foundation
for foundation in ['care', 'fairness', 'loyalty', 'authority', 'sanctity']:
    if foundation not in all_data:
        continue
    
    wass_dict = all_data[foundation]
    layers = sorted(wass_dict.keys())
    wasserstein = [wass_dict[l] for l in layers]
    
    ax.plot(
        layers,
        wasserstein,
        marker=MARKERS[foundation],
        markersize=8,
        linewidth=3,
        color=COLORS[foundation],
        label=foundation.capitalize(),
        alpha=0.85,
        markeredgecolor='white',
        markeredgewidth=1
    )

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.3)

# Add shaded regions for positive (correct) and negative (wrong) directions
y_min = min([min(all_data[f].values()) for f in all_data])
y_max = max([max(all_data[f].values()) for f in all_data])
y_range = y_max - y_min
y_min = y_min - 0.05 * y_range
y_max = y_max + 0.05 * y_range

ax.axhspan(0, y_max, alpha=0.05, color='green', zorder=0)
ax.axhspan(y_min, 0, alpha=0.05, color='red', zorder=0)

# Styling
ax.set_xlabel('Layer Index', fontsize=24, fontweight='bold')
ax.set_ylabel('Signed Wasserstein-1 Distance ($SW_1$)', fontsize=24, fontweight='bold')
# ax.set_title(
#     'Wasserstein Distance Across Layers (with Direction)\nMoral Foundations vs Nonmoral Content',
#     fontsize=24,
#     fontweight='bold',
#     pad=20
# )

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_axisbelow(True)

# Legend
ax.legend(
    loc='upper left',
    fontsize=24,
    framealpha=0.95,
    edgecolor='black',
    fancybox=True,
    shadow=False,
    ncol=1
)

# X-axis
ax.set_xticks(range(0, 32, 2))
ax.set_xlim(-0.5, 31.5)

# Y-axis
ax.set_ylim(y_min, y_max)

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Tick labels
ax.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n✅ Plot saved: {OUTPUT_PATH}")
print("="*70)