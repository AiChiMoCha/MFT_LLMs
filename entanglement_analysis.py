"""
Steering Impact Matrix: All 32 Layers

Generates a grid of 5x5 matrices for each layer (0-31).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ==================== Configuration ====================
RESULTS_FILES = {
    'Care': 'results/RB_care_vs_social_norms_all_layers.json',
    'Fairness': 'results/RB_fairness_vs_social_norms_all_layers.json',
    'Loyalty': 'results/RB_loyalty_vs_social_norms_all_layers.json',
    'Authority': 'results/RB_authority_vs_social_norms_all_layers.json',
    'Sanctity': 'results/RB_sanctity_vs_social_norms_all_layers.json'
}

FOUNDATIONS = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Sanctity']

KEY_MAP = {
    'Care': ['Care'],
    'Fairness': ['Fairness'],
    'Loyalty': ['Loyalty'],
    'Authority': ['Authority'],
    'Sanctity': ['Sanctity', 'Purity']
}

OUTPUT_DIR = './visualizations/'
NUM_LAYERS = 32


def find_json_key(data_dict, possible_keys):
    for key in possible_keys:
        if key in data_dict:
            return key
    return None


def get_slope_at_layer(by_layer_alpha, layer, foundation):
    """Calculate slope via linear regression"""
    layer_str = str(layer)
    if layer_str not in by_layer_alpha:
        return None
    
    alpha_dict = by_layer_alpha[layer_str]
    first_alpha = list(alpha_dict.keys())[0]
    json_key = find_json_key(alpha_dict[first_alpha], KEY_MAP[foundation])
    
    if json_key is None:
        return None
    
    alphas = []
    scores = []
    for alpha_str, foundation_scores in alpha_dict.items():
        if json_key in foundation_scores:
            alphas.append(float(alpha_str))
            scores.append(foundation_scores[json_key]['mean'])
    
    if len(alphas) < 3:
        return None
    
    slope, _, _, _, _ = stats.linregress(alphas, scores)
    return slope


# ==================== Main Analysis ====================
print("="*60)
print("Steering Impact Matrix: All 32 Layers")
print("="*60)

# Load all data
all_data = {}
for foundation in FOUNDATIONS:
    file_path = RESULTS_FILES[foundation]
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            all_data[foundation] = json.load(f)
        print(f"✓ Loaded {foundation}")
    else:
        print(f"⚠️ Missing: {file_path}")

# Build matrices for all layers
all_matrices = {}
global_max = 0

for layer in range(NUM_LAYERS):
    impact_matrix = np.zeros((5, 5))
    
    for i, steered in enumerate(FOUNDATIONS):
        if steered not in all_data:
            continue
        
        by_layer_alpha = all_data[steered]['summary_statistics']['by_layer_alpha']
        
        for j, measured in enumerate(FOUNDATIONS):
            slope = get_slope_at_layer(by_layer_alpha, layer, measured)
            if slope is not None:
                impact_matrix[i, j] = slope
    
    all_matrices[layer] = impact_matrix
    global_max = max(global_max, np.max(np.abs(impact_matrix)))

print(f"\nGlobal max |slope| = {global_max:.4f}")

# ==================== Visualization ====================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 8 rows x 4 cols = 32 subplots
fig, axes = plt.subplots(8, 4, figsize=(20, 36))
axes = axes.flatten()

for layer in range(NUM_LAYERS):
    ax = axes[layer]
    impact_matrix = all_matrices[layer]
    
    # Use global scale for comparison
    im = ax.imshow(impact_matrix, cmap='RdBu_r', aspect='equal',
                   vmin=-global_max, vmax=global_max)
    
    # Minimal labels
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['C', 'F', 'L', 'A', 'S'], fontsize=8)
    ax.set_yticklabels(['C', 'F', 'L', 'A', 'S'], fontsize=8)
    
    # Title with layer number
    diag_sum = sum(impact_matrix[i, i] for i in range(5))
    ax.set_title(f'L{layer} (Σdiag={diag_sum:.2f})', fontsize=10, fontweight='bold')
    
    # Add values
    for i in range(5):
        for j in range(5):
            value = impact_matrix[i, j]
            text_color = 'white' if abs(value) > 0.5 * global_max else 'black'
            fontsize = 7 if i == j else 6
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight=weight)
    
    # Highlight diagonal
    for i in range(5):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                    edgecolor='black', linewidth=1.5))

# Add colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Slope (Δscore / Δα)', fontsize=12)

plt.suptitle('Steering Impact Matrices Across All Layers\n'
             '(C=Care, F=Fairness, L=Loyalty, A=Authority, S=Sanctity)\n'
             'Rows=Steered, Cols=Measured, Diagonal=Target Effect',
             fontsize=14, fontweight='bold', y=0.995)

output_path = Path(OUTPUT_DIR) / 'steering_impact_all_layers.pdf'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Saved: {output_path}")

# ==================== Summary Statistics ====================
print("\n" + "="*60)
print("Layer Summary (sorted by diagonal sum)")
print("="*60)

layer_stats = []
for layer in range(NUM_LAYERS):
    m = all_matrices[layer]
    diag_sum = sum(m[i, i] for i in range(5))
    off_diag_sum = sum(abs(m[i, j]) for i in range(5) for j in range(5) if i != j)
    layer_stats.append((layer, diag_sum, off_diag_sum))

# Sort by diagonal sum descending
layer_stats.sort(key=lambda x: x[1], reverse=True)

print(f"{'Layer':<8} {'Σ Diagonal':<12} {'Σ |Off-diag|':<14} {'Ratio':<10}")
print("-"*50)
for layer, diag, off in layer_stats[:10]:
    ratio = off / diag if diag > 0.01 else float('inf')
    print(f"L{layer:<7} {diag:<12.3f} {off:<14.3f} {ratio:<10.2f}")

print("\n... (showing top 10 by target effect)")