"""
Ridge Plot - MAXIMUM SEPARATION EMPHASIS
Order: Care, Sanctity, Authority, Loyalty, Fairness (top to bottom)
Desaturated colors for softer appearance
Fairness uses Layer 30, others use Layer 31
Left and right labels aligned on Y-axis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.colors as mcolors

# ==================== Configuration ====================
BASE_DIR = "/data/cyu/MFT_LLM_ARR/projection_results_P1"

# Function to desaturate colors
def desaturate_color(hex_color, saturation_factor=0.6):
    """
    Reduce color saturation
    saturation_factor: 0-1, where 1 = original, 0 = grayscale
    """
    rgb = mcolors.hex2color(hex_color)
    hsv = mcolors.rgb_to_hsv(rgb)
    # Reduce saturation
    hsv[1] *= saturation_factor
    rgb_desat = mcolors.hsv_to_rgb(hsv)
    return mcolors.rgb2hex(rgb_desat)

# Original colors
ORIGINAL_COLORS = {
    'care': '#E91E63',
    'fairness': '#2196F3',
    'loyalty': '#4CAF50',
    'authority': '#FF9800',
    'sanctity': '#9C27B0'
}

# Desaturate colors (60% saturation)
COLORS = {k: desaturate_color(v, 0.6) for k, v in ORIGINAL_COLORS.items()}

# ⭐ W_1 values for each foundation
W1_VALUES = {
    'Care': 1.71,
    'Sanctity': 0.90,
    'Authority': 0.78,
    'Loyalty': 0.72,
    'Fairness': 0.082
}

# ⭐ Order from BOTTOM to TOP in the plot
# (Plot draws from bottom up, so reverse order)
# Format: (name, path, color, layer)
FOUNDATIONS = [
    ('Fairness', '/fairness_vs_social/fairness_vs_nonmoral_2000_20251225_222046/projections', COLORS['fairness'], 30),  # ⭐ Layer 30
    ('Loyalty', '/loyalty_vs_social/loyalty_vs_nonmoral_2000_20251225_222327/projections', COLORS['loyalty'], 31),
    ('Authority', '/authority_vs_social/authority_vs_nonmoral_2000_20251225_222910/projections', COLORS['authority'], 31),
    ('Sanctity', '/sanctity_vs_social/sanctity_vs_nonmoral_2000_20251225_223007/projections', COLORS['sanctity'], 31),
    ('Care', '/care_vs_social/care_vs_nonmoral_2000_20251225_221904/projections', COLORS['care'], 31),
]

BASELINE_NAME = "Nonmoral_2000"
OUTPUT_PATH = './ridge_plot_layer31_ordered_desaturated.pdf'

# ⭐⭐⭐ MAXIMUM SEPARATION PARAMETERS
X_MIN = -2.5
X_MAX = 3.5
BW_ADJUST = 0.25  # Very sharp peaks
OVERLAP = 0.15   # Minimal overlap
ALPHA = 0.45

# ⭐ Label alignment parameter
LABEL_Y_POSITION = 0.4  # Consistent Y position for both left and right labels

# ==================== Load Data ====================
print(f"Loading data...")
print("Order (top to bottom): Care, Sanctity, Authority, Loyalty, Fairness")
print("Layers: Care(31), Sanctity(31), Authority(31), Loyalty(31), Fairness(30)\n")

data_list = []

for foundation, path, color, layer in FOUNDATIONS:
    proj_dir = BASE_DIR + path
    
    f_path = f"{proj_dir}/layer_{layer}_{foundation}.npy"
    b_path = f"{proj_dir}/layer_{layer}_{BASELINE_NAME}.npy"
    
    f_raw = np.load(f_path)
    b_raw = np.load(b_path)
    
    mu = b_raw.mean()
    sigma = b_raw.std()
    f_std = (f_raw - mu) / sigma
    b_std = (b_raw - mu) / sigma
    
    data_list.append({
        'name': foundation,
        'layer': layer,
        'foundation': f_std,
        'nonmoral': b_std,
        'color': color,
        'median_f': np.median(f_std),
        'median_n': np.median(b_std),
        'separation': np.median(f_std) - np.median(b_std),
        'w1': W1_VALUES[foundation]  # ⭐ Add W1 value
    })
    
    print(f"  {foundation:10s} (L{layer}): Δ = {data_list[-1]['separation']:+6.3f}σ, W₁ = {W1_VALUES[foundation]:.3f}")

print(f"\n✅ Data loaded with desaturated colors\n")

# ==================== Create Plot ====================
fig, ax = plt.subplots(figsize=(18, 8))

x_range = np.linspace(X_MIN, X_MAX, 800)

# Calculate densities
max_density = 0
for data_info in data_list:
    kde_f = stats.gaussian_kde(data_info['foundation'], bw_method=BW_ADJUST)
    kde_n = stats.gaussian_kde(data_info['nonmoral'], bw_method=BW_ADJUST)
    max_density = max(max_density, kde_f(x_range).max(), kde_n(x_range).max())

ridge_height = 1.0
spacing = ridge_height * (1 - OVERLAP)
n_ridges = len(data_list)

# Plot ridges (from bottom to top)
for idx, data_info in enumerate(data_list):
    y_offset = idx * spacing
    
    kde_foundation = stats.gaussian_kde(data_info['foundation'], bw_method=BW_ADJUST)
    kde_nonmoral = stats.gaussian_kde(data_info['nonmoral'], bw_method=BW_ADJUST)
    
    density_foundation = kde_foundation(x_range)
    density_nonmoral = kde_nonmoral(x_range)
    
    density_foundation_norm = (density_foundation / max_density) * ridge_height
    density_nonmoral_norm = (density_nonmoral / max_density) * ridge_height
    
    # Nonmoral (very light gray)
    ax.fill_between(x_range, y_offset, y_offset + density_nonmoral_norm,
                    alpha=0.3, color='lightgray',
                    edgecolor='gray', linewidth=1, zorder=idx*2)
    
    # Foundation (desaturated color)
    ax.fill_between(x_range, y_offset, y_offset + density_foundation_norm,
                    alpha=ALPHA, color=data_info['color'],
                    edgecolor='white', linewidth=2, zorder=idx*2+1)
    
    # Foundation median
    ax.plot([data_info['median_f'], data_info['median_f']], 
           [y_offset, y_offset + ridge_height * 0.9],
           color=data_info['color'], linestyle='-', linewidth=4, alpha=1.0, zorder=idx*2+3)
    
    # ⭐ Calculate aligned Y position for labels
    label_y = y_offset + ridge_height * LABEL_Y_POSITION
    
    # ⭐ Left label - foundation name with W1 value
    label_text = f"{data_info['name']} ($SW_1={data_info['w1']:.2f}$)"
    ax.text(X_MIN - 0.35, label_y,
           label_text,
           fontsize=24, fontweight='bold', va='center', ha='right',
           color=data_info['color'])

# Styling
ax.set_xlim(X_MIN - 0.5, X_MAX + 0.8)
ax.set_ylim(-0.25, n_ridges * spacing + ridge_height * 0.15)

ax.set_xlabel('Standardized Projection Deviation',
              fontsize=24, fontweight='bold')

# ⭐ Set x-axis ticks with sigma labels
ax.set_xticks([-2, -1, 0, 1, 2, 3])
ax.set_xticklabels([r'$-2\sigma$', r'$-1\sigma$', r'$0$', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'], fontsize=20)

# Axes
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)

# Grid
ax.grid(axis='x', alpha=0.5, linestyle='-', linewidth=1, color='gray')
ax.set_axisbelow(True)

# Zero reference
ax.axvline(0, color='grey', linestyle='--', linewidth=2, alpha=0.6, zorder=1000)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f"✅ Ridge plot saved: {OUTPUT_PATH}")

# Statistics
print(f"\n{'='*70}")
print(f"Order (top to bottom):")
print(f"{'='*70}")
# Reverse order for display (top to bottom)
for i, data in enumerate(reversed(data_list), 1):
    print(f"{i}. {data['name']:10s} (Layer {data['layer']}, W₁={data['w1']:.3f}): {data['separation']:+6.3f}σ from baseline")
print(f"{'='*70}")