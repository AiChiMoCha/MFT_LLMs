"""
Optimal Layer Selection via Linear Regression Sensitivity
SAE Steering Results Analysis - 2 PDFs Only (with MMLU)

** Updates **
- Aligns with co-author's 'select_best_layer_sensitivity_qwen.py' normalization.
- Qwen alphas [-100, 100] are normalized by factor 50 to [-2, 2].
- Converts 1-based layer indices (from input JSON) to 0-based for visualization.
- Clean star labels (deduplicated) and correct titles.
- Legend placed inside the plot (upper left).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# ==================== Configuration ====================
# Single input file containing all foundations
appendix = 'qwen_top_10_range_100'
RESULTS_FILE = f'results/sae_steering_results_{appendix}.json'

OUTPUT_DIR = './visualizations/'
OUTPUT_PREFIX = f'sae_steering_sensitivity_{appendix}_normalized'

# Alpha normalization factor
# - Qwen:  α ∈ [-100, 100] → scale_factor = 50 → normalized to [-2, 2]
ALPHA_SCALE_FACTOR = 50
MODEL_NAME = "Qwen"  # For plot titles

FOUNDATION_MAP = {
    'care': 'Care',
    'fairness': 'Fairness',
    'loyalty': 'Loyalty',
    'authority': 'Authority',
    'sanctity': 'Sanctity'
}

# Possible JSON keys for each foundation in MFQ scores
FOUNDATION_JSON_KEYS = {
    'care': ['Care'],
    'fairness': ['Fairness'],
    'loyalty': ['Loyalty'],
    'authority': ['Authority'],
    'sanctity': ['Sanctity', 'Purity']
}

COLORS = {
    'Care': '#E91E63',
    'Fairness': '#2196F3',
    'Loyalty': '#4CAF50',
    'Authority': '#FF9800',
    'Sanctity': '#9C27B0'
}

# MMLU Performance Data - Load from results file
MMLU_RESULTS_FILE = 'results/sae_mmlu_steering_effect_qwen_top_10_range_100.json'

def normalize_alpha(alpha: float, scale_factor: float) -> float:
    """Normalize alpha value: α_normalized = α_original / scale_factor"""
    return alpha / scale_factor

def load_mmlu_data(filepath: str, scale_factor: float) -> dict:
    """
    Load MMLU steering results and compute average accuracy across foundations.
    Normalizes alphas by scale_factor.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get alpha range from config and normalize
    alphas_original = data['experiment_config']['alpha_range']
    alphas_norm = np.array([normalize_alpha(float(a), scale_factor) for a in alphas_original])
    
    # Compute average accuracy across all foundations for each alpha
    foundations = list(data['summary_statistics'].keys())
    
    accuracies_by_alpha = {str(float(a)): [] for a in alphas_original}
    
    for foundation in foundations:
        by_alpha = data['summary_statistics'][foundation]['by_alpha']
        for alpha in alphas_original:
            alpha_str = str(float(alpha))
            if alpha_str in by_alpha:
                # Convert to percentage
                accuracies_by_alpha[alpha_str].append(by_alpha[alpha_str]['accuracy'] * 100)
    
    # Average across foundations (ensure order matches normalized alphas)
    avg_accuracies = []
    for alpha in alphas_original:
        alpha_str = str(float(alpha))
        if accuracies_by_alpha[alpha_str]:
            avg_accuracies.append(np.mean(accuracies_by_alpha[alpha_str]))
        else:
            avg_accuracies.append(np.nan)
            
    return {
        'alphas': alphas_norm,
        'accuracy': np.array(avg_accuracies)
    }

# Load MMLU data
if os.path.exists(MMLU_RESULTS_FILE):
    MMLU_DATA = load_mmlu_data(MMLU_RESULTS_FILE, ALPHA_SCALE_FACTOR)
    print(f"✓ Loaded MMLU data from {MMLU_RESULTS_FILE} (Alphas normalized by {ALPHA_SCALE_FACTOR})")
else:
    print(f"⚠️ MMLU results file not found: {MMLU_RESULTS_FILE}, using placeholder")
    MMLU_DATA = {
        'alphas': np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
        'accuracy': np.array([63.7, 64.1, 64.8, 65.0, 65.2, 65.3, 64.8, 64.5, 64.1])
    }

# ==================== Helper Functions ====================
def find_foundation_key(score_dict, possible_keys):
    """Find which key actually exists in the JSON data"""
    for key in possible_keys:
        if key in score_dict:
            return key
    raise KeyError(f"None of {possible_keys} found in data. Available keys: {list(score_dict.keys())}")


def calculate_linear_sensitivity(alphas: List[float], scores: List[float]) -> Dict:
    """
    Calculate linear regression slope
    k_{f,l} = Cov(α, Score) / Var(α)
    """
    alphas_array = np.array(alphas)
    scores_array = np.array(scores)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(alphas_array, scores_array)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }


# ==================== Load Data and Calculate Sensitivity ====================
print("="*70)
print(f"Calculating Linear Regression Slope (SAE Steering) - {appendix}")
print(f"Alpha Normalization Factor: {ALPHA_SCALE_FACTOR}")
print("="*70)

# Load the single results file
with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

print(f"\n✓ Loaded: {RESULTS_FILE}")

foundations_in_data = list(data['results'].keys())

all_sensitivity = {}
all_alpha_scores = {}

for foundation_key in foundations_in_data:
    display_name = FOUNDATION_MAP.get(foundation_key, foundation_key.capitalize())
    foundation_data = data['results'][foundation_key]
    
    possible_keys = FOUNDATION_JSON_KEYS.get(foundation_key, [foundation_key.capitalize()])
    
    # Try to find the key from first available data point
    actual_json_key = None
    first_layer_str = list(foundation_data.keys())[0]
    first_alpha_str = list(foundation_data[first_layer_str].keys())[0]
    
    if 'mean_scores' in foundation_data[first_layer_str][first_alpha_str]:
        try:
            actual_json_key = find_foundation_key(
                foundation_data[first_layer_str][first_alpha_str]['mean_scores'],
                possible_keys
            )
        except KeyError:
            continue
    else:
        continue
    
    # Get all alpha values and Normalize them
    all_alphas_original = sorted([float(a) for a in foundation_data[first_layer_str].keys()])
    
    sensitivity_per_layer = {}
    alpha_scores_per_layer = {}
    
    # Iterate over layers (converting 1-based JSON keys to 0-based integers)
    for layer_str, layer_data in foundation_data.items():
        layer_idx_0_based = int(layer_str) - 1
        
        alphas_norm = []
        scores = []
        
        for alpha in all_alphas_original:
            alpha_key = str(int(alpha)) if alpha == int(alpha) else str(alpha)
            if alpha_key not in layer_data:
                 alpha_key = str(alpha)
            
            if alpha_key in layer_data:
                if 'mean_scores' in layer_data[alpha_key] and actual_json_key in layer_data[alpha_key]['mean_scores']:
                    alphas_norm.append(normalize_alpha(alpha, ALPHA_SCALE_FACTOR))
                    scores.append(layer_data[alpha_key]['mean_scores'][actual_json_key])
        
        if len(alphas_norm) < 3:
            continue
        
        regression_results = calculate_linear_sensitivity(alphas_norm, scores)
        
        sensitivity_per_layer[layer_idx_0_based] = regression_results
        alpha_scores_per_layer[layer_idx_0_based] = {
            'alphas': alphas_norm,
            'scores': scores
        }
    
    all_sensitivity[display_name] = sensitivity_per_layer
    all_alpha_scores[display_name] = alpha_scores_per_layer

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

foundations_ordered = list(all_sensitivity.keys())
optimal_layers = {}

for foundation_name in foundations_ordered:
    optimal_layer = max(all_sensitivity[foundation_name].keys(),
                       key=lambda l: all_sensitivity[foundation_name][l]['slope'])
    optimal_layers[foundation_name] = optimal_layer

# ==================== Figure 1: Slope Curves ====================
print("\nGenerating Figure 1: Slope Curves...")

fig1, ax1 = plt.subplots(figsize=(16, 10))

# 1. Plot all lines first
for foundation_name, sensitivity_dict in all_sensitivity.items():
    layers = sorted(sensitivity_dict.keys())
    slopes = [sensitivity_dict[l]['slope'] for l in layers]
    
    ax1.plot(layers, slopes, 'o-', linewidth=3.5, markersize=8,
            color=COLORS[foundation_name], label=foundation_name, alpha=0.85)

# 2. Plot stars and labels (Cleaned up)
optimal_points_by_layer = {}

for foundation_name, sensitivity_dict in all_sensitivity.items():
    opt_layer = optimal_layers[foundation_name]
    optimal_slope = sensitivity_dict[opt_layer]['slope']
    
    ax1.scatter(opt_layer, optimal_slope, s=500, marker='*',
               color=COLORS[foundation_name], edgecolors='black',
               linewidths=3, zorder=10)
    
    if opt_layer not in optimal_points_by_layer:
        optimal_points_by_layer[opt_layer] = []
    optimal_points_by_layer[opt_layer].append(optimal_slope)

# 3. Add text labels (only once per layer index)
for layer_idx, slopes in optimal_points_by_layer.items():
    ax1.text(layer_idx, max(slopes) + 0.03, f'L{layer_idx}',
            ha='center', va='bottom', fontsize=20, fontweight='bold',
            color='black')

ax1.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Layer Index (0-based)', fontsize=20, fontweight='bold')
ax1.set_ylabel(r'Slope $k_{f,l}$ (Normalized $\alpha$)', fontsize=20, fontweight='bold')

if ALPHA_SCALE_FACTOR != 1:
    ax1.set_title(f'{MODEL_NAME.upper()} - α normalized (÷{ALPHA_SCALE_FACTOR})', 
                  fontsize=18, fontweight='bold')

ax1.legend(fontsize=18, framealpha=0.95, edgecolor='black', loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')

all_layers_flat = [l for d in all_sensitivity.values() for l in d.keys()]
if all_layers_flat:
    ax1.set_xlim(min(all_layers_flat)-1, max(all_layers_flat)+1)

ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
output_path_1 = Path(OUTPUT_DIR) / f'{OUTPUT_PREFIX}_1_slope_curves.pdf'
plt.savefig(output_path_1, dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"✅ Saved: {output_path_1}")

# ==================== Figure 2: Delta Score Curves + MMLU ====================
print("Generating Figure 2: Delta Score Curves + MMLU...")

fig2, ax2 = plt.subplots(figsize=(16, 10))

foundation_plot_data = {}
all_deltas = []

for foundation_name in foundations_ordered:
    opt_layer = optimal_layers[foundation_name]
    alphas = np.array(all_alpha_scores[foundation_name][opt_layer]['alphas'], dtype=float)
    scores = all_alpha_scores[foundation_name][opt_layer]['scores']
    
    if 0.0 in alphas:
        baseline_score = scores[np.where(alphas == 0.0)[0][0]]
    else:
        baseline_score = all_sensitivity[foundation_name][opt_layer]['intercept']

    delta_scores = np.array([s - baseline_score for s in scores], dtype=float)
    
    foundation_plot_data[foundation_name] = {
        "opt_layer": opt_layer,
        "alphas": alphas,
        "delta": delta_scores,
        "slope": all_sensitivity[foundation_name][opt_layer]['slope'],
        "r2": all_sensitivity[foundation_name][opt_layer]['r_squared'],
    }
    all_deltas.extend(delta_scores.tolist())

slopes_abs = np.array([abs(v["slope"]) for v in foundation_plot_data.values()], dtype=float)
smin, smax = float(slopes_abs.min()), float(slopes_abs.max())
def lw_from_slope(s):
    if smax - smin < 1e-12: return 3.0
    return 2.0 + (abs(s) - smin) / (smax - smin) * 3.5

ax2.axhline(y=0, color='0.3', linestyle='-', linewidth=1.8, alpha=0.8, zorder=1)
ax2.axvline(x=0, color='0.5', linestyle=':', linewidth=1.6, alpha=0.8, zorder=1)

for foundation_name in foundations_ordered:
    d = foundation_plot_data[foundation_name]
    color = COLORS[foundation_name]
    lw = lw_from_slope(d["slope"])
    
    ax2.plot(d["alphas"], d["delta"], 'o', markersize=10,
             color=color, alpha=0.85, markeredgecolor='white', markeredgewidth=1.5,
             label=f'{foundation_name} (L{d["opt_layer"]})')
    
    alpha_range = np.linspace(float(d["alphas"].min()), float(d["alphas"].max()), 200)
    ax2.plot(alpha_range, d["slope"] * alpha_range, linestyle='-', linewidth=lw, color=color, alpha=0.95)

# MMLU Performance (Right Axis)
ax2_right = ax2.twinx()
ax2_right.plot(
    MMLU_DATA['alphas'], 
    MMLU_DATA['accuracy'],
    'o--', color='0.4', linewidth=3.0, markersize=9,
    markeredgecolor='white', markeredgewidth=1.5, alpha=0.85,
    label='MMLU Accuracy', zorder=5
)
ax2_right.set_ylabel('Average MMLU Accuracy (%)', fontsize=22, fontweight='bold')
ax2_right.tick_params(axis='y', labelsize=18)
ax2_right.spines['top'].set_visible(False)
ax2_right.set_ylim(0, 100)

# Axes Limits
all_alphas_arr = np.concatenate([d["alphas"] for d in foundation_plot_data.values()])
xmin, xmax = float(all_alphas_arr.min()), float(all_alphas_arr.max())
ax2.set_xlim(xmin - 0.2, xmax + 0.2)
max_abs_delta = float(np.max(np.abs(np.array(all_deltas, dtype=float)))) if all_deltas else 1.0
ax2.set_ylim(-max_abs_delta * 1.25, max_abs_delta * 1.25)

# Labels & Title
ax2.set_xlabel(r'Normalized Steering Strength ($\alpha$ / 50)', fontsize=22, fontweight='bold')
ax2.set_ylabel(r'$\Delta$ Score (relative to $\alpha=0$)', fontsize=22, fontweight='bold')
if ALPHA_SCALE_FACTOR != 1:
    ax2.set_title(f'{MODEL_NAME.upper()} - α normalized (÷{ALPHA_SCALE_FACTOR})', fontsize=18, fontweight='bold')

ax2.grid(True, alpha=0.25, linestyle='--')
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Legend (Upper Left, inside)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=18, 
          loc='upper left',
          framealpha=0.95, edgecolor='0.2', ncol=2)

# Stats Table
sorted_for_table = sorted(foundations_ordered, key=lambda f: foundation_plot_data[f]["slope"], reverse=True)
table_lines = [f"{'Foundation':<11s}  {'k':>6s}  {'R²':>5s}  {'Sig':>3s}"]
for f in sorted_for_table:
    opt_layer = foundation_plot_data[f]["opt_layer"]
    p_val = all_sensitivity[f][opt_layer]['p_value']
    sig_mark = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    table_lines.append(f"{f:<11s}  {foundation_plot_data[f]['slope']:+6.3f}  {foundation_plot_data[f]['r2']:5.3f}  {sig_mark:>3s}")

ax2.text(0.975, 0.05, "\n".join(table_lines), transform=ax2.transAxes, ha='right', va='bottom',
         fontsize=22, fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95))

plt.tight_layout()
output_path_2 = Path(OUTPUT_DIR) / f'{OUTPUT_PREFIX}_2_alpha_curves.pdf'
plt.savefig(output_path_2, dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"✅ Saved: {output_path_2}")

# Save Summary
summary_path = Path(OUTPUT_DIR) / f'{OUTPUT_PREFIX}_summary.txt'
with open(summary_path, 'w') as f:
    f.write(f"Steering Sensitivity Analysis - {MODEL_NAME} (Alpha Scale Factor: {ALPHA_SCALE_FACTOR})\n")
    f.write("="*80 + "\n")
    for f_name in foundations_ordered:
        opt_layer = optimal_layers[f_name]
        stats = all_sensitivity[f_name][opt_layer]
        f.write(f"{f_name:<12} L{opt_layer:<4} Slope: {stats['slope']:+.4f}\n")
print(f"✅ Summary saved: {summary_path}")