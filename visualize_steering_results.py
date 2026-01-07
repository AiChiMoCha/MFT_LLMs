"""
Visualize Moral Foundation Steering Results (Supports Non-MFQ Concepts)

Usage:
python visualize_steering_results_single.py \
    --results_path results/care_vs_social_norms_all_layers.json \
    --output_dir visualizations/care_vs_social_norms
"""

import json
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class SteeringVisualizerSingle:
    """Visualize steering experiment results - supports single MFQ concept"""
    
    # Foundation name mapping: concept_name -> MFQ_name
    FOUNDATION_NAME_MAP = {
        'care': 'Care',
        'fairness': 'Fairness', 
        'loyalty': 'Loyalty',
        'authority': 'Authority',
        'sanctity': 'Purity',
        'purity': 'Purity'
    }
    
    def __init__(self, results_path: str, output_dir: str):
        self.results_path = results_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(results_path, 'r') as f:
            self.data = json.load(f)
        
        self.config = self.data['experiment_config']
        self.summary = self.data['summary_statistics']
        
        self.concept_pair = self.config['concept_pair']
        foundation_A_raw, foundation_B_raw = self.concept_pair.split('_vs_')
        
        # Map concept names to MFQ names (may be None for non-MFQ concepts)
        self.foundation_A_concept = foundation_A_raw.capitalize()
        self.foundation_B_concept = foundation_B_raw.capitalize()
        
        self.foundation_A = self._map_foundation_name(foundation_A_raw)
        self.foundation_B = self._map_foundation_name(foundation_B_raw)
        
        print(f"Loaded results: {self.concept_pair}")
        total_tests = self.config.get('total_tests', self.config.get('successful_results', 'N/A'))
        print(f"Total tests: {total_tests}")
        print(f"Concept names: {self.foundation_A_concept} vs {self.foundation_B_concept}")
        print(f"MFQ mappings: {self.foundation_A} vs {self.foundation_B}")
        
        # Check which are MFQ concepts
        self.is_A_mfq = self.foundation_A is not None
        self.is_B_mfq = self.foundation_B is not None
        
        if not self.is_A_mfq and not self.is_B_mfq:
            raise ValueError("At least one concept must be an MFQ foundation!")
        
        print(f"A is MFQ: {self.is_A_mfq}, B is MFQ: {self.is_B_mfq}")
        
        # Show available foundations in data
        first_layer = list(self.summary['by_layer_alpha'].keys())[0]
        first_alpha = list(self.summary['by_layer_alpha'][first_layer].keys())[0]
        available_foundations = list(self.summary['by_layer_alpha'][first_layer][first_alpha].keys())
        print(f"Available foundations in data: {available_foundations}")
        
        print(f"Output: {self.output_dir}\n")
    
    def _map_foundation_name(self, concept_name: str) -> Optional[str]:
        """Map concept name to MFQ foundation name, return None if not MFQ concept"""
        # First try the hardcoded mapping
        mapped = self.FOUNDATION_NAME_MAP.get(concept_name.lower())
        if mapped:
            # Verify it exists in the data
            first_layer = list(self.summary['by_layer_alpha'].keys())[0]
            first_alpha = list(self.summary['by_layer_alpha'][first_layer].keys())[0]
            available = list(self.summary['by_layer_alpha'][first_layer][first_alpha].keys())
            
            if mapped in available:
                return mapped
        
        # If not in map or not in data, try to find by exact match
        first_layer = list(self.summary['by_layer_alpha'].keys())[0]
        first_alpha = list(self.summary['by_layer_alpha'][first_layer].keys())[0]
        available = list(self.summary['by_layer_alpha'][first_layer][first_alpha].keys())
        
        # Try exact match (case insensitive)
        for avail in available:
            if avail.lower() == concept_name.lower():
                return avail
        
        # Try capitalized version
        capitalized = concept_name.capitalize()
        if capitalized in available:
            return capitalized
        
        # If still not found, return None (non-MFQ concept)
        return None
    
    def create_all_visualizations(self):
        """Create all visualization plots"""
        print("Creating visualizations...")
        
        # 1. Heatmaps for all MFQ foundations
        self.plot_foundation_heatmaps()
        
        # 2. Target MFQ foundation(s) focus
        self.plot_target_foundation_focus()
        
        # 3. Per-layer alpha curves (all layers grid) for MFQ foundation(s)
        self.plot_all_layers_grid()
        
        # 4. Best layers analysis
        self.plot_best_layers()
        
        # 5. Alpha effect across layers
        self.plot_alpha_effect()
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}")
    
    def plot_foundation_heatmaps(self):
        """Heatmap for each MFQ foundation across layers and alphas"""
        
        by_layer_alpha = self.summary['by_layer_alpha']
        layers = sorted([int(k) for k in by_layer_alpha.keys()])
        alphas = self.config['alpha_range']
        
        # Get all foundation names
        first_layer = str(layers[0])
        first_alpha = str(alphas[0])
        foundations = list(by_layer_alpha[first_layer][first_alpha].keys())
        
        # Create subplot for each foundation
        n_foundations = len(foundations)
        n_cols = 3
        n_rows = (n_foundations + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        if n_foundations == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, foundation in enumerate(foundations):
            ax = axes[idx]
            
            # Build matrix
            matrix = np.zeros((len(layers), len(alphas)))
            for i, layer in enumerate(layers):
                for j, alpha in enumerate(alphas):
                    matrix[i, j] = by_layer_alpha[str(layer)][str(alpha)][foundation]['mean']
            
            # Plot heatmap
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=1, vmax=5)
            
            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f'{a:.1f}' for a in alphas], rotation=45)
            ax.set_xlabel('Alpha', fontsize=11)
            
            ax.set_yticks(range(0, len(layers), 4))
            ax.set_yticklabels([layers[i] for i in range(0, len(layers), 4)])
            ax.set_ylabel('Layer', fontsize=11)
            
            # Highlight target foundation(s)
            if self.is_A_mfq and foundation == self.foundation_A:
                title = f'{foundation} ({self.foundation_A_concept}) ★'
                ax.set_title(title, fontsize=13, fontweight='bold', color='red')
            elif self.is_B_mfq and foundation == self.foundation_B:
                title = f'{foundation} ({self.foundation_B_concept}) ★'
                ax.set_title(title, fontsize=13, fontweight='bold', color='blue')
            else:
                ax.set_title(foundation, fontsize=13)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Score', fontsize=10)
        
        # Remove empty subplots
        for idx in range(n_foundations, len(axes)):
            fig.delaxes(axes[idx])
        
        # Build title based on which concepts are MFQ
        if self.is_A_mfq and self.is_B_mfq:
            title = f'Foundation Scores Heatmaps: {self.foundation_A_concept} vs {self.foundation_B_concept}'
        elif self.is_A_mfq:
            title = f'Foundation Scores Heatmaps: {self.foundation_A_concept} (steered by {self.foundation_B_concept})'
        else:
            title = f'Foundation Scores Heatmaps: {self.foundation_B_concept} (steered by {self.foundation_A_concept})'
        
        plt.suptitle(
            f'{title}\nAll Layers × All Alphas',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        save_path = self.output_dir / 'foundation_heatmaps.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Foundation heatmaps: {save_path}")
    
    def plot_target_foundation_focus(self):
        """Focus on target MFQ foundation(s)"""
        
        by_layer_alpha = self.summary['by_layer_alpha']
        layers = sorted([int(k) for k in by_layer_alpha.keys()])
        alphas = self.config['alpha_range']
        
        # Determine which foundations to plot
        foundations_to_plot = []
        if self.is_A_mfq:
            foundations_to_plot.append((self.foundation_A, self.foundation_A_concept, 'red'))
        if self.is_B_mfq:
            foundations_to_plot.append((self.foundation_B, self.foundation_B_concept, 'blue'))
        
        n_plots = len(foundations_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
        if n_plots == 1:
            axes = [axes]
        
        for idx, (foundation, concept_name, color) in enumerate(foundations_to_plot):
            ax = axes[idx]
            
            # Build matrix
            matrix = np.zeros((len(layers), len(alphas)))
            for i, layer in enumerate(layers):
                for j, alpha in enumerate(alphas):
                    matrix[i, j] = by_layer_alpha[str(layer)][str(alpha)][foundation]['mean']
            
            # Plot heatmap
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=1, vmax=5)
            
            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=11, rotation=45)
            ax.set_xlabel('Alpha (Steering Strength)', fontsize=13, fontweight='bold')
            
            ax.set_yticks(range(0, len(layers), 2))
            ax.set_yticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=10)
            ax.set_ylabel('Layer', fontsize=13, fontweight='bold')
            
            # Get steering concept name
            if self.is_A_mfq and foundation == self.foundation_A:
                steering_concept = self.foundation_B_concept
            else:
                steering_concept = self.foundation_A_concept
            
            title = f'{foundation}\n({concept_name} steered by {steering_concept})'
            ax.set_title(title, fontsize=15, fontweight='bold', color=color)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('MFQ Score', fontsize=12)
        
        # Build subtitle
        if self.is_A_mfq and not self.is_B_mfq:
            subtitle = f'α > 0: Steer towards {self.foundation_A_concept} | α < 0: Steer towards {self.foundation_B_concept} (non-MFQ)'
        elif not self.is_A_mfq and self.is_B_mfq:
            subtitle = f'α > 0: Steer towards {self.foundation_A_concept} (non-MFQ) | α < 0: Steer towards {self.foundation_B_concept}'
        else:
            subtitle = f'α > 0: Steer towards {self.foundation_A_concept} | α < 0: Steer towards {self.foundation_B_concept}'
        
        plt.suptitle(
            f'Target Foundation Focus\n{subtitle}',
            fontsize=16, fontweight='bold'
        )
        
        plt.tight_layout()
        save_path = self.output_dir / 'target_focus.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Target focus: {save_path}")
    
    def plot_all_layers_grid(self):
        """Grid of alpha curves for all layers - only MFQ foundations"""
        
        by_layer_alpha = self.summary['by_layer_alpha']
        layers = sorted([int(k) for k in by_layer_alpha.keys()])
        alphas = self.config['alpha_range']
        
        # Determine which foundations to plot
        foundations_to_plot = []
        if self.is_A_mfq:
            foundations_to_plot.append((self.foundation_A, self.foundation_A_concept, 'red', 'o'))
        if self.is_B_mfq:
            foundations_to_plot.append((self.foundation_B, self.foundation_B_concept, 'blue', 's'))
        
        # Create grid
        n_layers = len(layers)
        n_cols = 8
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 3))
        axes = axes.flatten() if n_layers > 1 else [axes]
        
        for idx, layer in enumerate(layers):
            ax = axes[idx]
            
            # Plot each MFQ foundation
            for foundation, concept_name, color, marker in foundations_to_plot:
                scores = [by_layer_alpha[str(layer)][str(a)][foundation]['mean'] for a in alphas]
                ax.plot(alphas, scores, f'{marker}-', color=color, linewidth=2,
                       markersize=6, label=concept_name)
            
            # If both are MFQ, plot delta
            if len(foundations_to_plot) == 2:
                scores_A = [by_layer_alpha[str(layer)][str(a)][foundations_to_plot[0][0]]['mean'] for a in alphas]
                scores_B = [by_layer_alpha[str(layer)][str(a)][foundations_to_plot[1][0]]['mean'] for a in alphas]
                delta = [a - b for a, b in zip(scores_A, scores_B)]
                ax.plot(alphas, delta, '^--', color='purple', linewidth=1.5,
                       markersize=5, alpha=0.7, label='Δ (A-B)')
            
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_ylim(-2, 5)
            ax.grid(True, alpha=0.3)
            
            ax.set_title(f'Layer {layer}', fontsize=11, fontweight='bold')
            ax.set_xlabel('α', fontsize=9)
            
            if idx % n_cols == 0:
                ax.set_ylabel('Score', fontsize=9)
            
            if idx == 0:
                ax.legend(fontsize=8, loc='upper left')
        
        # Remove empty subplots
        for idx in range(n_layers, len(axes)):
            fig.delaxes(axes[idx])
        
        # Build title
        if self.is_A_mfq and not self.is_B_mfq:
            title = f'All Layers Alpha Curves: {self.foundation_A_concept} (MFQ) steered by {self.foundation_B_concept}'
        elif not self.is_A_mfq and self.is_B_mfq:
            title = f'All Layers Alpha Curves: {self.foundation_B_concept} (MFQ) steered by {self.foundation_A_concept}'
        else:
            title = f'All Layers Alpha Curves: {self.foundation_A_concept} vs {self.foundation_B_concept}'
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.998)
        
        plt.tight_layout()
        save_path = self.output_dir / 'all_layers_grid.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ All layers grid: {save_path}")
    
    def plot_best_layers(self):
        """Identify and plot best steering layers"""
        
        by_layer_alpha = self.summary['by_layer_alpha']
        layers = sorted([int(k) for k in by_layer_alpha.keys()])
        alphas = self.config['alpha_range']
        
        # Separate positive and negative alphas
        positive_alphas = [a for a in alphas if a > 0]
        negative_alphas = [a for a in alphas if a < 0]
        
        # Determine which foundation(s) to analyze
        if self.is_A_mfq and not self.is_B_mfq:
            # Only A is MFQ: measure change in A
            target_foundation = self.foundation_A
            target_name = self.foundation_A_concept
            steering_name = self.foundation_B_concept
        elif not self.is_A_mfq and self.is_B_mfq:
            # Only B is MFQ: measure change in B
            target_foundation = self.foundation_B
            target_name = self.foundation_B_concept
            steering_name = self.foundation_A_concept
        else:
            # Both MFQ: use original logic
            target_foundation = None
        
        # Calculate steering effectiveness for each layer
        layer_effectiveness_pos = {}
        layer_effectiveness_neg = {}
        
        if target_foundation:
            # Single MFQ foundation case
            if positive_alphas:
                for layer in layers:
                    baseline = by_layer_alpha[str(layer)]['0.0'][target_foundation]['mean']
                    max_pos_alpha = max(positive_alphas)
                    steered = by_layer_alpha[str(layer)][str(max_pos_alpha)][target_foundation]['mean']
                    delta = steered - baseline
                    layer_effectiveness_pos[layer] = {
                        'delta': delta,
                        'combined': abs(delta)
                    }
            
            if negative_alphas:
                for layer in layers:
                    baseline = by_layer_alpha[str(layer)]['0.0'][target_foundation]['mean']
                    min_neg_alpha = min(negative_alphas)
                    steered = by_layer_alpha[str(layer)][str(min_neg_alpha)][target_foundation]['mean']
                    delta = steered - baseline
                    layer_effectiveness_neg[layer] = {
                        'delta': delta,
                        'combined': abs(delta)
                    }
        else:
            # Both MFQ case: use original delta logic
            if positive_alphas:
                for layer in layers:
                    baseline_A = by_layer_alpha[str(layer)]['0.0'][self.foundation_A]['mean']
                    baseline_B = by_layer_alpha[str(layer)]['0.0'][self.foundation_B]['mean']
                    
                    max_pos_alpha = max(positive_alphas)
                    steered_A = by_layer_alpha[str(layer)][str(max_pos_alpha)][self.foundation_A]['mean']
                    steered_B = by_layer_alpha[str(layer)][str(max_pos_alpha)][self.foundation_B]['mean']
                    
                    delta_A = steered_A - baseline_A
                    delta_B = baseline_B - steered_B
                    
                    layer_effectiveness_pos[layer] = {
                        'delta_A': delta_A,
                        'delta_B': delta_B,
                        'combined': delta_A + delta_B
                    }
            
            if negative_alphas:
                for layer in layers:
                    baseline_A = by_layer_alpha[str(layer)]['0.0'][self.foundation_A]['mean']
                    baseline_B = by_layer_alpha[str(layer)]['0.0'][self.foundation_B]['mean']
                    
                    min_neg_alpha = min(negative_alphas)
                    steered_A = by_layer_alpha[str(layer)][str(min_neg_alpha)][self.foundation_A]['mean']
                    steered_B = by_layer_alpha[str(layer)][str(min_neg_alpha)][self.foundation_B]['mean']
                    
                    delta_B = steered_B - baseline_B
                    delta_A = baseline_A - steered_A
                    
                    layer_effectiveness_neg[layer] = {
                        'delta_A': delta_A,
                        'delta_B': delta_B,
                        'combined': delta_A + delta_B
                    }
        
        # Create figure
        n_groups = (1 if positive_alphas else 0) + (1 if negative_alphas else 0)
        if n_groups == 0:
            print("⚠️ No alpha values to plot")
            return
        
        fig, axes = plt.subplots(n_groups * 2, 5, figsize=(24, 10 * n_groups))
        if n_groups == 1:
            axes = axes.flatten()
        else:
            axes = axes.reshape(-1)
        
        current_idx = 0
        
        # Plot positive alpha effectiveness
        if positive_alphas:
            sorted_layers_pos = sorted(layer_effectiveness_pos.items(),
                                      key=lambda x: x[1]['combined'], reverse=True)
            
            for i in range(10):
                ax = axes[current_idx + i]
                
                if i >= len(sorted_layers_pos):
                    ax.axis('off')
                    continue
                
                layer, stats = sorted_layers_pos[i]
                
                if target_foundation:
                    # Single MFQ case
                    scores = [by_layer_alpha[str(layer)][str(a)][target_foundation]['mean'] for a in alphas]
                    ax.plot(alphas, scores, 'o-', color='red', linewidth=3, markersize=8, label=target_name)
                    title = f'Layer {layer} (Rank #{i+1})\nΔ: {stats["delta"]:+.2f}'
                else:
                    # Both MFQ case
                    scores_A = [by_layer_alpha[str(layer)][str(a)][self.foundation_A]['mean'] for a in alphas]
                    scores_B = [by_layer_alpha[str(layer)][str(a)][self.foundation_B]['mean'] for a in alphas]
                    
                    ax.plot(alphas, scores_A, 'o-', color='red', linewidth=3, markersize=8, label=self.foundation_A_concept)
                    ax.plot(alphas, scores_B, 's-', color='blue', linewidth=3, markersize=8, label=self.foundation_B_concept)
                    ax.fill_between(alphas, scores_A, scores_B, alpha=0.2, color='purple')
                    title = f'Layer {layer} (Rank #{i+1})\nΔ_A: {stats["delta_A"]:+.2f} | Δ_B: {stats["delta_B"]:+.2f}'
                
                ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
                ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
                ax.set_ylim(1, 5)
                ax.grid(True, alpha=0.3)
                
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel('α', fontsize=10)
                if i % 5 == 0:
                    ax.set_ylabel('Score', fontsize=10)
                if i == 0:
                    ax.legend(fontsize=9)
            
            current_idx += 10
            
            # Save ranking
            ranking_path_pos = self.output_dir / 'layer_ranking_positive.txt'
            with open(ranking_path_pos, 'w') as f:
                if target_foundation:
                    f.write(f"Layer Ranking for POSITIVE α (towards {self.foundation_A_concept})\n")
                    f.write(f"Measuring change in {target_name} (MFQ)\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"{'Rank':<6} {'Layer':<8} {'Δ':>12}\n")
                    f.write("-" * 70 + "\n")
                    for rank, (layer, stats) in enumerate(sorted_layers_pos, 1):
                        f.write(f"{rank:<6} {layer:<8} {stats['delta']:>+12.3f}\n")
                else:
                    f.write(f"Layer Ranking for POSITIVE α (towards {self.foundation_A_concept})\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"{'Rank':<6} {'Layer':<8} {'Δ_A':>10} {'Δ_B':>10} {'Combined':>12}\n")
                    f.write("-" * 70 + "\n")
                    for rank, (layer, stats) in enumerate(sorted_layers_pos, 1):
                        f.write(f"{rank:<6} {layer:<8} {stats['delta_A']:>+10.3f} "
                               f"{stats['delta_B']:>+10.3f} {stats['combined']:>+12.3f}\n")
        
        # Plot negative alpha effectiveness
        if negative_alphas:
            sorted_layers_neg = sorted(layer_effectiveness_neg.items(),
                                      key=lambda x: x[1]['combined'], reverse=True)
            
            for i in range(10):
                ax = axes[current_idx + i]
                
                if i >= len(sorted_layers_neg):
                    ax.axis('off')
                    continue
                
                layer, stats = sorted_layers_neg[i]
                
                if target_foundation:
                    # Single MFQ case
                    scores = [by_layer_alpha[str(layer)][str(a)][target_foundation]['mean'] for a in alphas]
                    ax.plot(alphas, scores, 'o-', color='blue', linewidth=3, markersize=8, label=target_name)
                    title = f'Layer {layer} (Rank #{i+1})\nΔ: {stats["delta"]:+.2f}'
                else:
                    # Both MFQ case
                    scores_A = [by_layer_alpha[str(layer)][str(a)][self.foundation_A]['mean'] for a in alphas]
                    scores_B = [by_layer_alpha[str(layer)][str(a)][self.foundation_B]['mean'] for a in alphas]
                    
                    ax.plot(alphas, scores_A, 'o-', color='red', linewidth=3, markersize=8, label=self.foundation_A_concept)
                    ax.plot(alphas, scores_B, 's-', color='blue', linewidth=3, markersize=8, label=self.foundation_B_concept)
                    ax.fill_between(alphas, scores_A, scores_B, alpha=0.2, color='purple')
                    title = f'Layer {layer} (Rank #{i+1})\nΔ_B: {stats["delta_B"]:+.2f} | Δ_A: {stats["delta_A"]:+.2f}'
                
                ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
                ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
                ax.set_ylim(1, 5)
                ax.grid(True, alpha=0.3)
                
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel('α', fontsize=10)
                if i % 5 == 0:
                    ax.set_ylabel('Score', fontsize=10)
                if i == 0:
                    ax.legend(fontsize=9)
            
            # Save ranking
            ranking_path_neg = self.output_dir / 'layer_ranking_negative.txt'
            with open(ranking_path_neg, 'w') as f:
                if target_foundation:
                    f.write(f"Layer Ranking for NEGATIVE α (towards {self.foundation_B_concept})\n")
                    f.write(f"Measuring change in {target_name} (MFQ)\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"{'Rank':<6} {'Layer':<8} {'Δ':>12}\n")
                    f.write("-" * 70 + "\n")
                    for rank, (layer, stats) in enumerate(sorted_layers_neg, 1):
                        f.write(f"{rank:<6} {layer:<8} {stats['delta']:>+12.3f}\n")
                else:
                    f.write(f"Layer Ranking for NEGATIVE α (towards {self.foundation_B_concept})\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"{'Rank':<6} {'Layer':<8} {'Δ_B':>10} {'Δ_A':>10} {'Combined':>12}\n")
                    f.write("-" * 70 + "\n")
                    for rank, (layer, stats) in enumerate(sorted_layers_neg, 1):
                        f.write(f"{rank:<6} {layer:<8} {stats['delta_B']:>+10.3f} "
                               f"{stats['delta_A']:>+10.3f} {stats['combined']:>+12.3f}\n")
        
        # Set title
        if target_foundation:
            if positive_alphas and negative_alphas:
                title = (f'Top 10 Most Effective Steering Layers for {target_name}\n'
                        f'Steered by {steering_name}')
            elif positive_alphas:
                title = (f'Top 10 Most Effective Steering Layers: Positive α\n'
                        f'{target_name} steered by {steering_name}')
            else:
                title = (f'Top 10 Most Effective Steering Layers: Negative α\n'
                        f'{target_name} steered by {steering_name}')
        else:
            if positive_alphas and negative_alphas:
                title = (f'Top 10 Most Effective Steering Layers\n'
                        f'Top: Positive α (↑{self.foundation_A_concept} + ↓{self.foundation_B_concept}) | '
                        f'Bottom: Negative α (↑{self.foundation_B_concept} + ↓{self.foundation_A_concept})')
            elif positive_alphas:
                title = (f'Top 10 Most Effective Steering Layers: Positive α\n'
                        f'↑{self.foundation_A_concept} + ↓{self.foundation_B_concept}')
            else:
                title = (f'Top 10 Most Effective Steering Layers: Negative α\n'
                        f'↑{self.foundation_B_concept} + ↓{self.foundation_A_concept}')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        save_path = self.output_dir / 'best_layers.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Best layers: {save_path}")
        if positive_alphas:
            print(f"✓ Positive α ranking: {ranking_path_pos}")
        if negative_alphas:
            print(f"✓ Negative α ranking: {ranking_path_neg}")
    
    def plot_alpha_effect(self):
        """Plot alpha effect averaged across all layers for MFQ foundations"""
        
        by_alpha = self.summary['by_alpha']
        alphas = sorted([float(k) for k in by_alpha.keys()])
        
        # Get all foundations
        foundations = list(by_alpha[str(alphas[0])].keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: All foundations
        ax1 = axes[0]
        for foundation in foundations:
            scores = [by_alpha[str(a)][foundation]['mean'] for a in alphas]
            stds = [by_alpha[str(a)][foundation]['std'] for a in alphas]
            
            if self.is_A_mfq and foundation == self.foundation_A:
                ax1.plot(alphas, scores, 'o-', linewidth=3, markersize=8,
                        label=f'{foundation} ({self.foundation_A_concept})', color='red')
                ax1.fill_between(alphas,
                                [s - std for s, std in zip(scores, stds)],
                                [s + std for s, std in zip(scores, stds)],
                                alpha=0.2, color='red')
            elif self.is_B_mfq and foundation == self.foundation_B:
                ax1.plot(alphas, scores, 's-', linewidth=3, markersize=8,
                        label=f'{foundation} ({self.foundation_B_concept})', color='blue')
                ax1.fill_between(alphas,
                                [s - std for s, std in zip(scores, stds)],
                                [s + std for s, std in zip(scores, stds)],
                                alpha=0.2, color='blue')
            else:
                ax1.plot(alphas, scores, '--', linewidth=1.5, markersize=5,
                        label=foundation, alpha=0.6)
        
        ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1.5)
        ax1.set_xlabel('Alpha (Steering Strength)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('MFQ Score (Averaged Across Layers)', fontsize=13, fontweight='bold')
        ax1.set_title('Alpha Effect: All Foundations', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(1, 5)
        
        # Right: Focus on target MFQ foundation(s)
        ax2 = axes[1]
        
        if self.is_A_mfq and self.is_B_mfq:
            # Both MFQ: show comparison
            scores_A = [by_alpha[str(a)][self.foundation_A]['mean'] for a in alphas]
            scores_B = [by_alpha[str(a)][self.foundation_B]['mean'] for a in alphas]
            stds_A = [by_alpha[str(a)][self.foundation_A]['std'] for a in alphas]
            stds_B = [by_alpha[str(a)][self.foundation_B]['std'] for a in alphas]
            delta = [a - b for a, b in zip(scores_A, scores_B)]
            
            ax2.plot(alphas, scores_A, 'o-', color='red', linewidth=3,
                    markersize=10, label=f'{self.foundation_A_concept} (A)')
            ax2.fill_between(alphas,
                            [s - std for s, std in zip(scores_A, stds_A)],
                            [s + std for s, std in zip(scores_A, stds_A)],
                            alpha=0.2, color='red')
            
            ax2.plot(alphas, scores_B, 's-', color='blue', linewidth=3,
                    markersize=10, label=f'{self.foundation_B_concept} (B)')
            ax2.fill_between(alphas,
                            [s - std for s, std in zip(scores_B, stds_B)],
                            [s + std for s, std in zip(scores_B, stds_B)],
                            alpha=0.2, color='blue')
            
            ax2_twin = ax2.twinx()
            ax2_twin.plot(alphas, delta, '^--', color='purple', linewidth=2.5,
                         markersize=8, label='Δ (A - B)', alpha=0.8)
            ax2_twin.axhline(y=0, color='gray', linestyle=':', linewidth=1)
            ax2_twin.set_ylabel('Score Difference (Δ)', fontsize=12, fontweight='bold')
            ax2_twin.legend(fontsize=10, loc='lower right')
            
            title = f'{self.foundation_A_concept} vs {self.foundation_B_concept}'
        else:
            # One MFQ: show single foundation
            if self.is_A_mfq:
                target = self.foundation_A
                target_name = self.foundation_A_concept
                steering_name = self.foundation_B_concept
                color = 'red'
                marker = 'o'
            else:
                target = self.foundation_B
                target_name = self.foundation_B_concept
                steering_name = self.foundation_A_concept
                color = 'blue'
                marker = 's'
            
            scores = [by_alpha[str(a)][target]['mean'] for a in alphas]
            stds = [by_alpha[str(a)][target]['std'] for a in alphas]
            
            ax2.plot(alphas, scores, f'{marker}-', color=color, linewidth=3,
                    markersize=10, label=target_name)
            ax2.fill_between(alphas,
                            [s - std for s, std in zip(scores, stds)],
                            [s + std for s, std in zip(scores, stds)],
                            alpha=0.2, color=color)
            
            title = f'{target_name} (MFQ) steered by {steering_name}'
        
        ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1.5)
        ax2.set_xlabel('Alpha (Steering Strength)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('MFQ Score', fontsize=13, fontweight='bold')
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(1, 5)
        
        # Build subtitle
        if self.is_A_mfq and not self.is_B_mfq:
            subtitle = f'α > 0: towards {self.foundation_A_concept} | α < 0: towards {self.foundation_B_concept} (non-MFQ)'
        elif not self.is_A_mfq and self.is_B_mfq:
            subtitle = f'α > 0: towards {self.foundation_A_concept} (non-MFQ) | α < 0: towards {self.foundation_B_concept}'
        else:
            subtitle = f'α > 0: towards {self.foundation_A_concept} | α < 0: towards {self.foundation_B_concept}'
        
        plt.suptitle(
            f'Alpha Effect Averaged Across All Layers\n{subtitle}',
            fontsize=16, fontweight='bold'
        )
        
        plt.tight_layout()
        save_path = self.output_dir / 'alpha_effect.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Alpha effect: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Steering Results (Supports Non-MFQ Concepts)')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = SteeringVisualizerSingle(
        results_path=args.results_path,
        output_dir=args.output_dir
    )
    
    # Generate all plots
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()