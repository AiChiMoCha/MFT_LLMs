"""
Optimal Layer Selection via Linear Regression Sensitivity
Cross-Foundation Analysis - 2 PDFs Only

Mathematical Definition:
k_{f,l} = Cov(α, Score) / Var(α)  (Linear regression slope)
L* = argmax_l(k_{f,l})  (Select layer with maximum positive slope)

Usage:
python select_best_layer_sensitivity.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# ==================== Configuration ====================
RESULTS_FILES = {
    'care': 'results/RB_care_vs_social_norms_all_layers.json',
    'fairness': 'results/RB_fairness_vs_social_norms_all_layers.json',
    'loyalty': 'results/RB_loyalty_vs_social_norms_all_layers.json',
    'authority': 'results/RB_authority_vs_social_norms_all_layers.json',
    'sanctity': 'results/RB_sanctity_vs_social_norms_all_layers.json'
}

OUTPUT_DIR = './visualizations/'
OUTPUT_PREFIX = 'optimal_layer_sensitivity'

FOUNDATION_MAP = {
    'care': 'Care',
    'fairness': 'Fairness',
    'loyalty': 'Loyalty',
    'authority': 'Authority',
    'sanctity': 'Sanctity'
}

# Possible JSON keys for each foundation
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

# MMLU Performance Data
MMLU_DATA = {
    'alphas': np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
    'accuracy': np.array([63.7, 64.1, 64.8, 65.0, 65.2, 65.3, 64.8, 64.5, 64.1])
}

# ==================== Helper Functions ====================
def find_foundation_key(alpha_dict, possible_keys):
    """Find which key actually exists in the JSON data"""
    for key in possible_keys:
        if key in alpha_dict:
            return key
    raise KeyError(f"None of {possible_keys} found in data. Available keys: {list(alpha_dict.keys())}")


def calculate_linear_sensitivity(alphas: List[float], scores: List[float]) -> Dict:
    """
    Calculate linear regression slope
    
    k_{f,l} = Cov(α, Score) / Var(α)
    
    Returns:
        dict with slope, intercept, r_value, p_value, std_err
    """
    alphas_array = np.array(alphas)
    scores_array = np.array(scores)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(alphas_array, scores_array)
    
    return {
        'slope': slope,  # k_{f,l}
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }


# ==================== Load Data and Calculate Sensitivity ====================
print("="*70)
print("Calculating Linear Regression Slope")
print("Mathematical Definition:")
print("  k_{f,l} = Cov(α, Score) / Var(α)")
print("  L* = argmax_l(k_{f,l})  (Maximum positive slope)")
print("="*70)

all_sensitivity = {}
all_alpha_scores = {}  # Store alpha-score pairs for plotting

for foundation_key, file_path in RESULTS_FILES.items():
    print(f"\nProcessing {foundation_key.upper()}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    by_layer_alpha = data['summary_statistics']['by_layer_alpha']
    display_name = FOUNDATION_MAP[foundation_key]
    
    # Find the actual key used in JSON
    first_layer = list(by_layer_alpha.keys())[0]
    first_alpha_key = list(by_layer_alpha[first_layer].keys())[0]
    possible_keys = FOUNDATION_JSON_KEYS[foundation_key]
    
    try:
        actual_json_key = find_foundation_key(
            by_layer_alpha[first_layer][first_alpha_key], 
            possible_keys
        )
        print(f"  Using JSON key: '{actual_json_key}'")
    except KeyError as e:
        print(f"  ⚠️ Skipping: {e}")
        continue
    
    # Get all alpha values
    all_alphas = sorted([float(a) for a in by_layer_alpha[first_layer].keys()])
    print(f"  Alpha values: {all_alphas}")
    
    sensitivity_per_layer = {}
    alpha_scores_per_layer = {}
    
    for layer_str, alpha_dict in by_layer_alpha.items():
        layer = int(layer_str)
        
        # Collect (alpha, score) pairs
        alphas = []
        scores = []
        
        for alpha in all_alphas:
            alpha_str = str(alpha)
            if alpha_str in alpha_dict and actual_json_key in alpha_dict[alpha_str]:
                alphas.append(alpha)
                scores.append(alpha_dict[alpha_str][actual_json_key]['mean'])
        
        if len(alphas) < 3:  # Need at least 3 points for regression
            continue
        
        # Calculate linear regression
        regression_results = calculate_linear_sensitivity(alphas, scores)
        
        sensitivity_per_layer[layer] = regression_results
        alpha_scores_per_layer[layer] = {
            'alphas': alphas,
            'scores': scores
        }
    
    all_sensitivity[display_name] = sensitivity_per_layer
    all_alpha_scores[display_name] = alpha_scores_per_layer
    
    # Find optimal layer (maximum slope, not absolute)
    if sensitivity_per_layer:
        optimal_layer = max(sensitivity_per_layer.keys(),
                           key=lambda l: sensitivity_per_layer[l]['slope'])
        optimal_stats = sensitivity_per_layer[optimal_layer]
        
        print(f"  L* = {optimal_layer}")
        print(f"  Slope k = {optimal_stats['slope']:+.4f}")
        print(f"  R² = {optimal_stats['r_squared']:.4f}")

print("\n" + "="*70)
print("Summary")
print("="*70)

if not all_sensitivity:
    print("❌ No valid data found!")
    exit(1)

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

foundations_ordered = list(all_sensitivity.keys())
optimal_layers = {}

# Calculate optimal layers (max slope, not abs)
for foundation_name in foundations_ordered:
    optimal_layer = max(all_sensitivity[foundation_name].keys(),
                       key=lambda l: all_sensitivity[foundation_name][l]['slope'])
    optimal_layers[foundation_name] = optimal_layer

first_foundation = foundations_ordered[0]
all_layers = sorted(list(all_sensitivity[first_foundation].keys()))

# ==================== Figure 1: Slope Curves (with sign) ====================
print("\nGenerating Figure 1: Slope Curves...")

fig1, ax1 = plt.subplots(figsize=(16, 10))

for foundation_name, sensitivity_dict in all_sensitivity.items():
    layers = sorted(sensitivity_dict.keys())
    slopes = [sensitivity_dict[l]['slope'] for l in layers]  # Keep sign
    
    ax1.plot(layers, slopes, 'o-', linewidth=3.5, markersize=8,
            color=COLORS[foundation_name], label=foundation_name, alpha=0.85)
    
    opt_layer = optimal_layers[foundation_name]
    optimal_slope = sensitivity_dict[opt_layer]['slope']
    
    ax1.scatter(opt_layer, optimal_slope, s=500, marker='*',
               color=COLORS[foundation_name], edgecolors='black',
               linewidths=3, zorder=10)
    ax1.text(opt_layer, optimal_slope + 0.03, f'L{opt_layer}',
            ha='center', va='bottom', fontsize=20, fontweight='bold',
            color=COLORS[foundation_name])

# Add zero reference line
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)

ax1.set_xlabel('Layer Index', fontsize=20, fontweight='bold')
ax1.set_ylabel(r'Slope $k_{f,l}$ (with sign)', fontsize=20, fontweight='bold')
ax1.legend(fontsize=18, framealpha=0.95, edgecolor='black', loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(-0.5, 31.5)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
output_path_1 = Path(OUTPUT_DIR) / f'{OUTPUT_PREFIX}_1_slope_curves.pdf'
plt.savefig(output_path_1, dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"✅ Saved: {output_path_1}")

# ==================== Figure 2: Delta Score Curves at Optimal Layers + MMLU ====================
print("Generating Figure 2: Delta Score Curves at Optimal Layers + MMLU Performance...")

fig2, ax2 = plt.subplots(figsize=(16, 10))

# ---------- options ----------
EMPHASIZE_POSITIVE_ONLY = False   # True: only emphasize α>=0
SHOW_NEGATIVE_POINTS = True      # True: also plot α<0 points (but lighter)
USE_SYMMETRIC_YLIM = True        # True: symmetric y-axis; False: focus on positive
Y_POS_FOCUS = True              # If USE_SYMMETRIC_YLIM=False, this decides positive range

# ---------- compute per-foundation deltas & slopes ----------
foundation_plot_data = {}
all_deltas = []

for foundation_name in foundations_ordered:
    opt_layer = optimal_layers[foundation_name]

    alphas = all_alpha_scores[foundation_name][opt_layer]['alphas']
    scores = all_alpha_scores[foundation_name][opt_layer]['scores']

    # baseline score at alpha=0 (if missing, use intercept)
    if 0.0 in alphas:
        baseline_score = scores[alphas.index(0.0)]
    else:
        baseline_score = all_sensitivity[foundation_name][opt_layer]['intercept']

    delta_scores = np.array([s - baseline_score for s in scores], dtype=float)

    regression = all_sensitivity[foundation_name][opt_layer]
    slope = float(regression['slope'])
    r2 = float(regression['r_squared'])

    foundation_plot_data[foundation_name] = {
        "opt_layer": opt_layer,
        "alphas": np.array(alphas, dtype=float),
        "delta": delta_scores,
        "slope": slope,
        "r2": r2,
    }
    all_deltas.extend(delta_scores.tolist())

# ---------- map slope to linewidth (bigger slope => thicker line) ----------
slopes_abs = np.array([abs(v["slope"]) for v in foundation_plot_data.values()], dtype=float)
smin, smax = float(slopes_abs.min()), float(slopes_abs.max())
def lw_from_slope(s):
    if smax - smin < 1e-12:
        return 3.0
    return 2.0 + (abs(s) - smin) / (smax - smin) * (5.5 - 2.0)

# ---------- plot foundation curves ----------
# Make baseline lines subtle
ax2.axhline(y=0, color='0.3', linestyle='-', linewidth=1.8, alpha=0.8, zorder=1)
ax2.axvline(x=0, color='0.5', linestyle=':', linewidth=1.6, alpha=0.8, zorder=1)

# plot each foundation
for foundation_name in foundations_ordered:
    d = foundation_plot_data[foundation_name]
    opt_layer = d["opt_layer"]
    alphas = d["alphas"]
    delta_scores = d["delta"]
    slope = d["slope"]
    r2 = d["r2"]

    color = COLORS[foundation_name]
    lw = lw_from_slope(slope)

    # split by sign of alpha
    pos_mask = alphas >= 0
    neg_mask = alphas < 0

    # points: α>=0 emphasized
    ax2.plot(
        alphas[pos_mask], delta_scores[pos_mask],
        'o', markersize=10,
        color=color, alpha=0.85,
        markeredgecolor='white', markeredgewidth=1.5,
        label=f'{foundation_name} (L{opt_layer})'
    )

    # points: α<0 de-emphasized (optional)
    if SHOW_NEGATIVE_POINTS and np.any(neg_mask):
        ax2.plot(
            alphas[neg_mask], delta_scores[neg_mask],
            'o', markersize=8,
            color=color, alpha=0.25,
            markeredgecolor='white', markeredgewidth=1.2
        )

    # regression line as a ray starting at 0
    if EMPHASIZE_POSITIVE_ONLY:
        alpha_max = float(alphas.max())
        alpha_range = np.linspace(0.0, alpha_max, 120)
    else:
        alpha_range = np.linspace(float(alphas.min()), float(alphas.max()), 200)

    reg_line_delta = slope * alpha_range  # Delta Score = k * alpha

    ax2.plot(
        alpha_range, reg_line_delta,
        linestyle='-',
        linewidth=lw,
        color=color,
        alpha=0.95
    )

# ---------- Add MMLU performance on secondary y-axis ----------
ax2_right = ax2.twinx()

# Plot MMLU data with dashed line
ax2_right.plot(
    MMLU_DATA['alphas'], 
    MMLU_DATA['accuracy'],
    'o--',
    color='0.4',
    linewidth=3.0,
    markersize=9,
    markeredgecolor='white',
    markeredgewidth=1.5,
    alpha=0.85,
    label='MMLU Accuracy',
    zorder=5
)

# Set y-axis label for MMLU
ax2_right.set_ylabel('Average MMLU Accuracy (%)', fontsize=22, fontweight='bold')
ax2_right.tick_params(axis='y', labelsize=18)
ax2_right.spines['top'].set_visible(False)

# Set y-axis limits for MMLU (0-100%)
ax2_right.set_ylim(0, 100)

# ---------- axes limits for left y-axis ----------
all_alphas = np.concatenate([foundation_plot_data[f]["alphas"] for f in foundations_ordered])
xmin = float(all_alphas.min())
xmax = float(all_alphas.max())

if EMPHASIZE_POSITIVE_ONLY:
    ax2.set_xlim(-0.1, xmax + 0.15)
else:
    ax2.set_xlim(xmin - 0.2, xmax + 0.2)

# y range for delta scores
max_abs_delta = float(np.max(np.abs(np.array(all_deltas, dtype=float)))) if all_deltas else 1.0
pad = 1.25

if USE_SYMMETRIC_YLIM:
    ax2.set_ylim(-max_abs_delta * pad, max_abs_delta * pad)
else:
    if Y_POS_FOCUS:
        ax2.set_ylim(-0.05 * max_abs_delta, max_abs_delta * pad)
    else:
        ax2.set_ylim(-max_abs_delta * pad, 0.05 * max_abs_delta)

# ---------- labels ----------
ax2.set_xlabel(r'Steering Strength ($\alpha$)', fontsize=22, fontweight='bold')
ax2.set_ylabel(r'$\Delta$ Score (relative to $\alpha=0$)', fontsize=22, fontweight='bold')

# ---------- grid & spines ----------
ax2.grid(True, alpha=0.25, linestyle='--')
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ---------- legend (combine both axes) ----------
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=22, loc='upper left', 
          framealpha=0.95, edgecolor='0.2')

# ---------- compact stats table on the right (remove L* column) ----------
sorted_for_table = sorted(
    foundations_ordered,
    key=lambda f: foundation_plot_data[f]["slope"],
    reverse=True
)

# Table without L* column
table_lines = [f"{'Foundation':<11s}  {'k':>6s}  {'R²':>5s}  {'Sig':>3s}"]
for f in sorted_for_table:
    opt_layer = foundation_plot_data[f]["opt_layer"]
    k = foundation_plot_data[f]["slope"]
    r2 = foundation_plot_data[f]["r2"]
    
    # Get p-value and determine significance
    p_val = all_sensitivity[f][opt_layer]['p_value']
    if p_val < 0.001:
        sig_mark = "***"
    elif p_val < 0.01:
        sig_mark = "**"
    elif p_val < 0.05:
        sig_mark = "*"
    else:
        sig_mark = "ns"
    
    table_lines.append(f"{f:<11s}  {k:+6.3f}  {r2:5.3f}  {sig_mark:>3s}")

table_text = "\n".join(table_lines)
ax2.text(
    0.975, 0.05,
    table_text,
    transform=ax2.transAxes,
    ha='right', va='bottom',
    fontsize=22,
    fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.3', alpha=0.95)
)

plt.tight_layout()
output_path_2 = Path(OUTPUT_DIR) / f'{OUTPUT_PREFIX}_2_alpha_curves.pdf'
plt.savefig(output_path_2, dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"✅ Saved: {output_path_2}")


# ==================== Save Summary Table ====================
summary_path = Path(OUTPUT_DIR) / f'{OUTPUT_PREFIX}_summary.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Linear Regression Slope Analysis - Summary\n")
    f.write("="*80 + "\n\n")
    
    f.write("Mathematical Definition:\n")
    f.write("  k_{f,l} = Cov(α, Score) / Var(α)  (Linear regression slope)\n")
    f.write("  L* = argmax_l(k_{f,l})  (Layer with maximum positive slope)\n")
    f.write("  Delta Score = Score - Score(α=0)  (Change from baseline)\n\n")
    
    f.write("="*80 + "\n")
    f.write("Optimal Layers by Foundation (Sorted by Slope)\n")
    f.write("="*80 + "\n\n")
    
    # Sort by slope (descending)
    sorted_foundations = sorted(foundations_ordered,
                               key=lambda f: all_sensitivity[f][optimal_layers[f]]['slope'],
                               reverse=True)
    
    f.write(f"{'Rank':<6} {'Foundation':<12} {'L*':<6} {'Slope k':<12} {'R²':<10}\n")
    f.write("-"*80 + "\n")
    
    for rank, foundation_name in enumerate(sorted_foundations, 1):
        opt_layer = optimal_layers[foundation_name]
        stats = all_sensitivity[foundation_name][opt_layer]
        f.write(f"{rank:<6} {foundation_name:<12} {opt_layer:<6} {stats['slope']:<+12.4f} "
               f"{stats['r_squared']:<10.4f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("Detailed Layer-by-Layer Results\n")
    f.write("="*80 + "\n\n")
    
    for foundation_name in foundations_ordered:
        opt_layer = optimal_layers[foundation_name]
        f.write(f"\n{foundation_name}:\n")
        f.write(f"{'Layer':<8} {'Slope k':<12} {'R²':<10} "
               f"{'Intercept':<12} {'p-value':<12}\n")
        f.write("-"*80 + "\n")
        
        # Sort layers by slope (descending) for each foundation
        sorted_layers = sorted(all_sensitivity[foundation_name].keys(),
                             key=lambda l: all_sensitivity[foundation_name][l]['slope'],
                             reverse=True)
        
        for layer in sorted_layers:
            stats = all_sensitivity[foundation_name][layer]
            marker = " ★" if layer == opt_layer else ""
            f.write(f"{layer:<8} {stats['slope']:<+12.4f} "
                   f"{stats['r_squared']:<10.4f} {stats['intercept']:<12.4f} "
                   f"{stats['p_value']:<12.4e}{marker}\n")

print(f"✅ Summary saved: {summary_path}")

print("\n" + "="*70)
print("✅ All Visualizations Complete!")
print("="*70)
print("\nGenerated files:")
print(f"  1. {output_path_1}")
print(f"  2. {output_path_2}")
print(f"  3. {summary_path}")
print("\nOptimal Layers (sorted by slope):")
sorted_foundations = sorted(foundations_ordered,
                           key=lambda f: all_sensitivity[f][optimal_layers[f]]['slope'],
                           reverse=True)
for rank, foundation_name in enumerate(sorted_foundations, 1):
    opt_layer = optimal_layers[foundation_name]
    opt_slope = all_sensitivity[foundation_name][opt_layer]['slope']
    print(f"  {rank}. {foundation_name:10s}: L{opt_layer:2d}, k={opt_slope:+.4f}")
print("="*70)