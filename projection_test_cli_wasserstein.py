"""
概念向量投影测试脚本 - SOTA版本 (Empirical-Distribution & Wasserstein-based)
改进要点：
1) 以两样本 Wasserstein 距离（W-D）衡量分布可分性，替代 Cohen's d / KS-D 作为主筛选/排序指标
2) 采用经验分布展示（直方图 + CDF），并在 CDF 上标注 Wasserstein 区域（填充表示分布偏移量）
3) 多层同时检验时，提供 Benjamini–Hochberg (FDR) 校正的 p_adj
4) 自适应 bin（Freedman–Diaconis）与可配置阈值 (W_min, p_max)
5) 右下角图：-log10(p-value) 显著性趋势
6) ⭐ 方向判定：确保 group1 在向量正方向（concept1 方向）
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

# ================== 统计与工具函数 ==================

try:
    from scipy.stats import wasserstein_distance, mannwhitneyu
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def freedman_diaconis_bins(x: np.ndarray, max_bins: int = 100) -> int:
    """Freedman-Diaconis规则计算最优bin数"""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 10
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0:
        return min(max_bins, max(10, int(np.sqrt(n))))
    h = 2 * iqr * (n ** (-1/3))
    if h <= 0:
        return min(max_bins, max(10, int(np.sqrt(n))))
    bins = int(np.ceil((x.max() - x.min()) / h))
    if bins <= 0:
        bins = 10
    return int(min(max_bins, max(10, bins)))

def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """经验累积分布函数"""
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def wasserstein_test(a: np.ndarray, b: np.ndarray, n_bootstrap: int = 500) -> Dict[str, Any]:
    """两样本 Wasserstein 距离 + bootstrap p 值估计"""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {"W": np.nan, "p": np.nan, "x_star": 0.0, "Fa_star": 0.0, "Fb_star": 0.0}
    if SCIPY_OK:
        W = float(wasserstein_distance(a, b))
    else:
        # 手动积分近似
        xa = np.sort(a)
        xb = np.sort(b)
        grid = np.linspace(min(xa.min(), xb.min()), max(xa.max(), xb.max()), 500)
        Fa = np.searchsorted(xa, grid, side="right") / len(xa)
        Fb = np.searchsorted(xb, grid, side="right") / len(xb)
        W = float(np.trapz(np.abs(Fa - Fb), grid))
    # bootstrap 近似 p-值
    obs = np.concatenate([a, b])
    n1 = len(a)
    rng = np.random.default_rng(42)
    null_vals = []
    for _ in range(n_bootstrap):
        rng.shuffle(obs)
        a_b = obs[:n1]
        b_b = obs[n1:]
        if SCIPY_OK:
            null_vals.append(wasserstein_distance(a_b, b_b))
        else:
            xa_b = np.sort(a_b)
            xb_b = np.sort(b_b)
            grid_b = np.linspace(min(xa_b.min(), xb_b.min()), max(xa_b.max(), xb_b.max()), 500)
            Fa_b = np.searchsorted(xa_b, grid_b, side="right") / len(xa_b)
            Fb_b = np.searchsorted(xb_b, grid_b, side="right") / len(xb_b)
            null_vals.append(np.trapz(np.abs(Fa_b - Fb_b), grid_b))
    null_vals = np.array(null_vals)
    p_boot = float(np.mean(null_vals >= W))
    x_star = (np.mean(a) + np.mean(b)) / 2
    return {"W": W, "p": p_boot, "x_star": x_star, "Fa_star": 0.5, "Fb_star": 0.5}

def auc_common_language(a: np.ndarray, b: np.ndarray) -> float:
    """Common Language Effect Size (AUC)"""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if SCIPY_OK:
        u = mannwhitneyu(a, b, alternative="two-sided").statistic
        return float(u / (len(a) * len(b)))
    rng = np.random.default_rng(123)
    A = a if len(a) <= 5000 else rng.choice(a, 5000, replace=False)
    B = b if len(b) <= 5000 else rng.choice(b, 5000, replace=False)
    count = 0
    ties = 0
    for xv in A:
        count += np.sum(xv > B)
        ties  += np.sum(xv == B)
    return float((count + 0.5 * ties) / (len(A) * len(B)))

def bh_correction(pvals: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR校正"""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    p_adj = np.empty_like(adj)
    p_adj[order] = adj
    reject = p_adj <= alpha
    return p_adj, reject

# ================== 主类及流程 ==================

class ProjectionTester:
    """概念向量投影测试器"""
    def __init__(self, model_path: str, vector_dir: str, device: str = "cuda"):
        self.model_path = model_path
        self.vector_dir = vector_dir
        self.device = device
        print("="*70)
        print("初始化投影测试器 (带方向判定)")
        print("="*70)
        print(f"模型路径: {model_path}")
        print(f"向量目录: {vector_dir}")
        print(f"设备: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map="auto" if device != "cpu" else None
        ).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ 模型加载完成")
        self.activations = []
        self.hook = None

    def load_concept_vector(self, layer_idx: int):
        vector_files = os.listdir(self.vector_dir)
        target_file = None
        for f in vector_files:
            if f"layers_{layer_idx}_residual_stream" in f and 'normalized.npy' in f:
                target_file = f
                break
        if target_file is None:
            for f in vector_files:
                if f"layers_{layer_idx}" in f and 'normalized.npy' in f:
                    target_file = f
                    break
        if target_file is None:
            raise FileNotFoundError(f"找不到第{layer_idx}层的向量文件")
        vector_path = os.path.join(self.vector_dir, target_file)
        print(f"  加载向量: {target_file}")
        vector = np.load(vector_path)
        vector_tensor = torch.tensor(vector, dtype=torch.float32 if self.device == "cpu" else torch.float16).to(self.device)
        print(f"  向量维度: {vector_tensor.shape[0]}, 范数: {torch.norm(vector_tensor.float()).item():.4f}")
        return vector_tensor, target_file

    def _hook_fn(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        last_token_activation = hidden_states[:, -1, :].detach()
        self.activations.append(last_token_activation)

    def compute_projections(self, texts: List[str], layer_idx: int, concept_vector: torch.Tensor):
        target_module = self.model.model.layers[layer_idx]
        self.hook = target_module.register_forward_hook(self._hook_fn)
        self.activations = []
        projections = []
        print(f"  计算 {len(texts)} 条文本的 projection（层 {layer_idx}）…")
        for text in tqdm(texts, desc=f"Layer {layer_idx}", ncols=80):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            activation = self.activations[-1]
            projection = torch.dot(activation.flatten(), concept_vector.flatten()).item()
            projections.append(projection)
        self.hook.remove()
        self.hook = None
        return np.array(projections)

def extract_vector_name(vector_dir: str) -> str:
    """从向量目录路径中提取向量名称"""
    path_parts = Path(vector_dir).parts
    for part in reversed(path_parts):
        if '_vs_' in part.lower():
            try:
                parts = part.lower().split('_')
                vs_idx = parts.index('vs')
                if vs_idx > 0 and vs_idx < len(parts) - 1:
                    return f"{parts[vs_idx-1]}_vs_{parts[vs_idx+1]}"
            except:
                pass
    return Path(vector_dir).parent.name

def create_output_directory(vector_dir: str, g1: str, g2: str, base_dir: str = "./projection_results") -> str:
    """创建智能命名的输出目录"""
    vn = extract_vector_name(vector_dir)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    dn = f"{g1.lower()}_vs_{g2.lower()}_{ts}"
    od = os.path.join(base_dir, vn, dn)
    os.makedirs(od, exist_ok=True)
    os.makedirs(os.path.join(od, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(od, "projections"), exist_ok=True)
    return od

def load_json_scenarios(json_path: str) -> List[str]:
    """加载JSON格式的测试数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['scenario'] for item in data]

def calculate_statistics(g1: np.ndarray, g2: np.ndarray) -> Dict[str, Any]:
    """计算统计量：Wasserstein、Cohen's d、median、IQR、方向等
    
    ⭐ 新增方向判定：
    - 计算 g1 和 g2 的中位数/均值
    - 判断 g1 是否在 g2 的右侧（正向）
    """
    wass = wasserstein_test(g1, g2, n_bootstrap=500)
    auc = auc_common_language(g1, g2)
    
    # 计算均值和中位数
    g1_mean = g1.mean()
    g2_mean = g2.mean()
    g1_median = np.median(g1)
    g2_median = np.median(g2)
    
    mean_diff = g1_mean - g2_mean
    median_diff = g1_median - g2_median
    
    pooled = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    cohen_d = mean_diff / pooled if pooled > 0 else 0.0
    
    # ⭐ 方向判定
    # 正确方向: group1 应该在 group2 的右侧（正方向）
    # 即: median(g1) > median(g2) 或 mean(g1) > mean(g2)
    correct_direction_median = g1_median > g2_median
    correct_direction_mean = g1_mean > g2_mean
    
    # 综合判定：优先用中位数（更稳健），如果中位数一致则看均值
    if abs(median_diff) > 1e-6:  # 中位数有差异
        correct_direction = correct_direction_median
    else:  # 中位数几乎一致，看均值
        correct_direction = correct_direction_mean
    
    q1_g1, q3_g1 = np.percentile(g1, [25, 75])
    q1_g2, q3_g2 = np.percentile(g2, [25, 75])
    
    return {
        'group1': {
            'n': len(g1),
            'mean': float(g1_mean),
            'std': float(g1.std()),
            'median': float(g1_median),
            'iqr': float(q3_g1 - q1_g1),
            'q1': float(q1_g1),
            'q3': float(q3_g1),
            'min': float(g1.min()),
            'max': float(g1.max())
        },
        'group2': {
            'n': len(g2),
            'mean': float(g2_mean),
            'std': float(g2.std()),
            'median': float(g2_median),
            'iqr': float(q3_g2 - q1_g2),
            'q1': float(q1_g2),
            'q3': float(q3_g2),
            'min': float(g2.min()),
            'max': float(g2.max())
        },
        'separation': {
            'mean_difference': float(mean_diff),
            'median_difference': float(median_diff),
            'cohen_d': float(cohen_d)
        },
        'direction': {
            'correct_direction': bool(correct_direction),
            'correct_direction_median': bool(correct_direction_median),
            'correct_direction_mean': bool(correct_direction_mean),
            'median_diff': float(median_diff),
            'mean_diff': float(mean_diff)
        },
        'wass': {
            'D': wass['W'],
            'p': wass['p'],
            'x_star': wass['x_star'],
            'Fa_star': wass['Fa_star'],
            'Fb_star': wass['Fb_star']
        },
        'auc': float(auc)
    }

def visualize_single_layer_empirical(g1: np.ndarray, g2: np.ndarray, layer_idx: int,
                                     save_path: str, g1_name: str, g2_name: str, 
                                     wass_info: Dict[str, float], direction_info: Dict[str, Any]):
    """单层详细可视化：empirical distribution + CDF + Wasserstein area + 方向标注"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    W_D = wass_info['D']
    p_val = wass_info['p']
    correct_dir = direction_info['correct_direction']
    median_diff = direction_info['median_diff']
    
    # 标题带方向信息
    dir_symbol = "✅" if correct_dir else "❌"
    dir_text = f"Direction: {dir_symbol} {'Correct' if correct_dir else 'WRONG'}"
    
    fig.suptitle(f'Layer {layer_idx}: {g1_name} vs {g2_name} | W-D={W_D:.3f}, p={p_val:.1e}\n{dir_text}',
                 fontsize=17, fontweight='bold', y=0.98)
    
    all_data = np.concatenate([g1, g2])
    min_val, max_val = all_data.min(), all_data.max()
    margin = (max_val - min_val)*0.1
    
    # ===== 1. 主图：Empirical Distribution =====
    ax_main = fig.add_subplot(gs[0, :])
    n_bins = freedman_diaconis_bins(all_data)
    bins = np.linspace(min_val-margin, max_val+margin, n_bins)
    
    ax_main.hist(g2, bins=bins, alpha=0.85, label=g2_name, color='#FF9966', edgecolor='white', linewidth=0.5)
    ax_main.hist(g1, bins=bins, alpha=0.85, label=g1_name, color='#6699CC', edgecolor='white', linewidth=0.5)
    
    ax_main.axvline(np.median(g2), color='#CC6633', linestyle='--', linewidth=2.5, label=f'{g2_name} Median')
    ax_main.axvline(np.median(g1), color='#336699', linestyle='--', linewidth=2.5, label=f'{g1_name} Median')
    ax_main.axvline(0, color='gray', linestyle=':', linewidth=2, alpha=0.6)
    
    y_max = ax_main.get_ylim()[1]
    arrow_y = y_max * 0.92
    text_y = y_max * 0.98
    
    # 方向箭头和标注
    ax_main.annotate('', xy=(min_val-margin*0.5, arrow_y),
                    xytext=(min_val+(max_val-min_val)*0.35, arrow_y),
                    arrowprops=dict(arrowstyle='<-', lw=3, color='#CC6633'))
    ax_main.text(min_val+(max_val-min_val)*0.15, text_y, f'← {g2_name} Direction',
                fontsize=13, fontweight='bold', color='#CC6633', ha='center', va='top')
    
    ax_main.annotate('', xy=(max_val+margin*0.5, arrow_y),
                    xytext=(max_val-(max_val-min_val)*0.35, arrow_y),
                    arrowprops=dict(arrowstyle='->', lw=3, color='#336699'))
    ax_main.text(max_val-(max_val-min_val)*0.15, text_y, f'{g1_name} Direction →',
                fontsize=13, fontweight='bold', color='#336699', ha='center', va='top')
    
    ax_main.set_xlabel('Projection Value', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Frequency (Count)', fontsize=13, fontweight='bold')
    
    # 标题中加入方向判定结果
    title_color = 'green' if correct_dir else 'red'
    ax_main.set_title(f"Empirical Distribution | {dir_text}", 
                     fontsize=14, fontweight='bold', pad=20, color=title_color)
    
    ax_main.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_main.grid(alpha=0.3, linestyle='--')
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # ===== 2. 归一化直方图 =====
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(g2, bins=bins, alpha=0.7, label=g2_name, color='#FF9966', edgecolor='white', density=True, linewidth=0.5)
    ax2.hist(g1, bins=bins, alpha=0.7, label=g1_name, color='#6699CC', edgecolor='white', density=True, linewidth=0.5)
    ax2.axvline(np.median(g2), color='#CC6633', linestyle='--', linewidth=2)
    ax2.axvline(np.median(g1), color='#336699', linestyle='--', linewidth=2)
    ax2.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Projection Value', fontsize=11)
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.set_title('Normalized Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ===== 3. CDF with Wasserstein Area =====
    ax3 = fig.add_subplot(gs[1, 1])
    sg2, yg2 = _ecdf(g2)
    sg1, yg1 = _ecdf(g1)
    ax3.plot(sg2, yg2, label=g2_name, linewidth=2.2, color='#FF9966')
    ax3.plot(sg1, yg1, label=g1_name, linewidth=2.2, color='#6699CC')
    grid = np.linspace(min_val-margin, max_val+margin, 300)
    Fa_interp = np.interp(grid, sg1, yg1, left=0, right=1)
    Fb_interp = np.interp(grid, sg2, yg2, left=0, right=1)
    ax3.fill_between(grid, Fa_interp, Fb_interp, color='red', alpha=0.15, label='Wasserstein area')
    ax3.set_xlabel('Projection Value', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('CDF with Wasserstein Area', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ===== 4. Box Plot =====
    ax4 = fig.add_subplot(gs[2, 0])
    bp = ax4.boxplot([g2, g1], labels=[g2_name, g1_name], patch_artist=True, notch=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#FF9966')
    bp['boxes'][1].set_facecolor('#6699CC')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    for i, data in enumerate([g2, g1], 1):
        y = data
        x = np.random.normal(i, 0.04, size=len(y))
        ax4.plot(x, y, 'o', alpha=0.3, markersize=4, color='black')
    ax4.set_ylabel('Projection Value', fontsize=11)
    ax4.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # ===== 5. Statistics Summary =====
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    dir_status = "✅ CORRECT" if correct_dir else "❌ WRONG"
    dir_explanation = f"{g1_name} is {'>' if median_diff > 0 else '<'} {g2_name}"
    
    summary = f"""
Statistical Summary (Layer {layer_idx})
{'='*40}

{g1_name}:
  Median:  {np.median(g1):8.3f}
  IQR:     {np.percentile(g1,75)-np.percentile(g1,25):8.3f}
  Range:   [{g1.min():.2f}, {g1.max():.2f}]
  N:       {len(g1)}

{g2_name}:
  Median:  {np.median(g2):8.3f}
  IQR:     {np.percentile(g2,75)-np.percentile(g2,25):8.3f}
  Range:   [{g2.min():.2f}, {g2.max():.2f}]
  N:       {len(g2)}

Separation:
  Wasserstein D: {W_D:8.3f}
  p-value:       {p_val:8.1e}
  Median Diff:   {median_diff:8.3f}

Direction Check:
  Status:        {dir_status}
  Explanation:   {dir_explanation}
  Expected:      {g1_name} > {g2_name}
"""
    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ 可视化保存: {save_path}")

def visualize_all_layers_summary(all_results: Dict[int,Dict[str,Any]],
                                 save_path: str, group1_name: str, group2_name: str):
    """汇总图：四个子图，右下角显示 -log10(p-value)"""
    layers = sorted(all_results.keys())
    W_ds = [all_results[l]['statistics']['wass']['D'] for l in layers]
    pvals = [all_results[l]['statistics']['wass']['p'] for l in layers]
    g1_means = [all_results[l]['statistics']['group1']['mean'] for l in layers]
    g2_means = [all_results[l]['statistics']['group2']['mean'] for l in layers]
    mean_diffs = [all_results[l]['statistics']['separation']['mean_difference'] for l in layers]
    
    # 方向信息
    correct_dirs = [all_results[l]['statistics']['direction']['correct_direction'] for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'All Layers Summary: {group1_name} vs {group2_name} Projection', 
                 fontsize=16, fontweight='bold')

    # ===== 1. 左上：Wasserstein D 趋势（带方向标记） =====
    ax1 = axes[0, 0]
    # 根据方向正确性设置颜色
    colors_wd = ['#2E7D32' if cd else '#D32F2F' for cd in correct_dirs]
    for i, (layer, wd, color) in enumerate(zip(layers, W_ds, colors_wd)):
        ax1.plot(layer, wd, marker='o', markersize=8, color=color, 
                markeredgecolor='black', markeredgewidth=0.5)
    ax1.plot(layers, W_ds, linestyle='-', linewidth=1.5, color='gray', alpha=0.3, zorder=0)
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Wasserstein D', fontsize=12)
    ax1.set_title('Wasserstein D Across Layers\n(Green=Correct Dir, Red=Wrong Dir)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ===== 2. 右上：均值趋势 =====
    ax2 = axes[0, 1]
    ax2.plot(layers, g1_means, marker='o', linewidth=2, markersize=6, 
            color='#336699', label=group1_name)
    ax2.plot(layers, g2_means, marker='o', linewidth=2, markersize=6, 
            color='#CC6633', label=group2_name)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Mean Projection', fontsize=12)
    ax2.set_title('Mean Projections Across Layers', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ===== 3. 左下：均值差（带方向标记） =====
    ax3 = axes[1, 0]
    # 根据方向正确性和均值差的符号设置颜色
    colors_diff = []
    for cd, md in zip(correct_dirs, mean_diffs):
        if cd and md > 0:
            colors_diff.append('#2E7D32')  # 正确且正向：绿色
        elif not cd and md < 0:
            colors_diff.append('#D32F2F')  # 错误且负向：红色
        else:
            colors_diff.append('#FF9800')  # 其他情况：橙色
    
    bars = ax3.bar(layers, mean_diffs, alpha=0.7, edgecolor='black', linewidth=0.8, color=colors_diff)
    ax3.axhline(0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Layer Index', fontsize=12)
    ax3.set_ylabel(f'Mean Difference ({group1_name} - {group2_name})', fontsize=12)
    ax3.set_title('Mean Difference Across Layers\n(Green=Correct, Red=Wrong, Orange=Mixed)', 
                 fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ===== 4. 右下：-log10(p-value) 显著性趋势 =====
    ax4 = axes[1, 1]
    
    min_p = 1e-10
    log_pvals = []
    for p in pvals:
        if p <= 0 or np.isnan(p):
            log_pvals.append(-np.log10(min_p))
        else:
            log_pvals.append(-np.log10(max(p, min_p)))
    
    # 颜色编码（结合方向判定）
    colors = []
    for lp, cd in zip(log_pvals, correct_dirs):
        if not cd:  # 方向错误
            colors.append('red')  # 不管显著性如何，方向错就是红色
        elif lp >= 1.3:  # 显著且方向正确
            colors.append('green')
        elif lp >= 1.0:  # 接近显著且方向正确
            colors.append('orange')
        else:  # 不显著
            colors.append('lightcoral')
    
    bars = ax4.bar(layers, log_pvals, alpha=0.75, edgecolor='black', 
                   linewidth=0.8, color=colors)
    
    ax4.axhline(y=1.3, color='darkgreen', linestyle='--', linewidth=2, 
               label='α = 0.05 (p = 0.05)', alpha=0.7)
    ax4.axhline(y=1.0, color='darkorange', linestyle=':', linewidth=1.5, 
               label='p = 0.1', alpha=0.5)
    
    # 标注最显著的几个层（只标注方向正确的）
    top_n = 5
    valid_indices = [i for i, cd in enumerate(correct_dirs) if cd]
    if valid_indices:
        valid_log_pvals = [log_pvals[i] for i in valid_indices]
        valid_layers = [layers[i] for i in valid_indices]
        sorted_valid = sorted(zip(valid_log_pvals, valid_layers, valid_indices), reverse=True)
        
        for lp, layer_idx, orig_idx in sorted_valid[:top_n]:
            if lp >= 1.3:
                ax4.text(layer_idx, lp + 0.15, f'{layer_idx}', 
                        ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    
    ax4.set_xlabel('Layer Index', fontsize=12)
    ax4.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax4.set_title('-log₁₀(p-value) Across Layers\n(Red=Wrong Direction, regardless of significance)', 
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_ylim(0, max(log_pvals) * 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 汇总图保存: {save_path}")

def generate_report(all_results: Dict[int,Dict[str,Any]], output_dir: str,
                    group1_name: str, group2_name: str, args: argparse.Namespace):
    """生成测试报告（带BH校正和方向判定）"""
    layers = sorted(all_results.keys())
    pvals = [all_results[l]['statistics']['wass']['p'] for l in layers]
    p_adj, reject = bh_correction(pvals, alpha=args.bh_alpha)
    
    for i, l in enumerate(layers):
        all_results[l]['statistics']['wass']['p_adj'] = float(p_adj[i])
        all_results[l]['statistics']['wass']['reject_bh'] = bool(reject[i])

    report_lines = []
    report_lines.append("="*100)
    report_lines.append(f"{group1_name} vs {group2_name} 概念向量投影测试报告（Wasserstein-based + 方向判定）")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*100)
    
    report_lines.append("\n【测试配置】")
    report_lines.append(f"模型: {args.model_path}")
    report_lines.append(f"向量目录: {args.vector_path}")
    report_lines.append(f"测试数据1: {args.test_file1} ({group1_name})")
    report_lines.append(f"测试数据2: {args.test_file2} ({group2_name})")
    report_lines.append(f"测试层: {layers}")
    report_lines.append(f"Wasserstein判定: D>={args.wass_d_min} 且 p<={args.wass_p_max}")
    report_lines.append(f"方向判定: {group1_name} 应在 {group2_name} 的正方向（右侧）")
    report_lines.append(f"多重校正: BH-FDR α={args.bh_alpha}")
    report_lines.append(f"输出目录: {output_dir}")
    
    report_lines.append("\n【方向判定说明】")
    report_lines.append(f"向量定义: {group1_name} - {group2_name}")
    report_lines.append(f"预期: median({group1_name}) > median({group2_name})")
    report_lines.append(f"判定: 如果方向错误（{group1_name}在左侧），该层标记为 FAIL")
    
    report_lines.append("\n【各层结果（⭐ 新增方向列）】")
    header = f"\n{'层':<6}{'W-D':>10}{'p':>14}{'p_adj':>14}{'-log10(p)':>12}{'方向':>8}{'综合判定':>12}"
    report_lines.append(header)
    report_lines.append("-"*100)
    
    passed_layers = []
    for l in layers:
        st = all_results[l]['statistics']
        D = st['wass']['D']
        p = st['wass']['p']
        padj = st['wass']['p_adj']
        log_p = -np.log10(max(p, 1e-10)) if p > 0 else 10.0
        
        # 方向判定
        correct_dir = st['direction']['correct_direction']
        dir_symbol = "✅" if correct_dir else "❌"
        
        # ⭐ 综合判定：必须同时满足分离度、显著性和方向
        separation_ok = (D >= args.wass_d_min) and (p <= args.wass_p_max)
        direction_ok = correct_dir
        
        # 只有两者都满足才 PASS
        ok = separation_ok and direction_ok
        
        if separation_ok and not direction_ok:
            flag = "FAIL(DIR)"  # 分离度够，但方向错
        elif not separation_ok and direction_ok:
            flag = "FAIL(SEP)"  # 方向对，但分离度不够
        elif not separation_ok and not direction_ok:
            flag = "FAIL(BOTH)" # 都不满足
        else:
            flag = "PASS"       # 都满足
        
        if ok:
            passed_layers.append(l)
        
        report_lines.append(f"{l:<6}{D:>10.3f}{p:>14.1e}{padj:>14.1e}{log_p:>12.2f}{dir_symbol:>8}{flag:>12}")
    
    # 最佳层（只在方向正确的层中选择）
    correct_dir_layers = [l for l in layers if all_results[l]['statistics']['direction']['correct_direction']]
    
    if correct_dir_layers:
        best_layer = max(correct_dir_layers, key=lambda k: all_results[k]['statistics']['wass']['D'])
        best_D = all_results[best_layer]['statistics']['wass']['D']
        best_p = all_results[best_layer]['statistics']['wass']['p']
        best_log_p = -np.log10(max(best_p, 1e-10)) if best_p > 0 else 10.0
        
        report_lines.append(f"\n🏆 最佳层(按W-D, 仅方向正确): 第 {best_layer} 层")
        report_lines.append(f"   W-D = {best_D:.3f}, p = {best_p:.1e}, -log10(p) = {best_log_p:.2f}")
    else:
        report_lines.append(f"\n⚠️ 警告: 没有方向正确的层！")
        report_lines.append(f"   这可能表示：")
        report_lines.append(f"   1. 向量方向定义错误（{group1_name} - {group2_name} vs 实际训练）")
        report_lines.append(f"   2. 测试数据与向量训练数据不匹配")
        report_lines.append(f"   3. 概念向量未能有效捕获概念差异")
    
    report_lines.append(f"\n✅ 通过综合判定的层数: {len(passed_layers)}/{len(layers)}")
    if passed_layers:
        report_lines.append(f"   层号: {passed_layers}")
    
    # 方向统计
    n_correct = sum(1 for l in layers if all_results[l]['statistics']['direction']['correct_direction'])
    n_wrong = len(layers) - n_correct
    report_lines.append(f"\n📊 方向统计:")
    report_lines.append(f"   方向正确: {n_correct}/{len(layers)} ({100*n_correct/len(layers):.1f}%)")
    report_lines.append(f"   方向错误: {n_wrong}/{len(layers)} ({100*n_wrong/len(layers):.1f}%)")
    
    if n_wrong > n_correct:
        report_lines.append(f"\n⚠️ 警告: 多数层方向错误！请检查向量定义和测试数据是否匹配。")
    
    report_lines.append("\n" + "="*100)
    
    with open(os.path.join(output_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='概念向量投影测试工具 - SOTA版（Wasserstein + 方向判定）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python %(prog)s --model_path /path/to/model --vector_path /path/to/vectors \\
      --test_file1 group1.json --test_file2 group2.json \\
      --wass_d_min 0.3 --wass_p_max 0.05

方向判定说明:
  向量定义为 concept1 - concept2 时:
  - test_file1 应该是 concept1 的数据
  - test_file2 应该是 concept2 的数据
  - 正确方向: median(group1) > median(group2)
  
  例如: authority_vs_social_norms 向量
  - test_file1 = authority.json
  - test_file2 = social_norm.json
  - 期望: authority projection > social_norm projection
        """
    )
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--vector_path', type=str, required=True, help='向量目录路径')
    parser.add_argument('--test_file1', type=str, required=True, help='第一组测试数据')
    parser.add_argument('--test_file2', type=str, required=True, help='第二组测试数据')
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cuda','cpu'], 
                       help='运行设备')
    parser.add_argument('--wass_d_min', type=float, default=0.5, 
                       help='单层判定的最小 Wasserstein-D（默认：0.5）')
    parser.add_argument('--wass_p_max', type=float, default=0.01, 
                       help='单层判定的最大 p 值（默认：0.01）')
    parser.add_argument('--bh_alpha', type=float, default=0.05, 
                       help='多重假设检验 FDR α（默认：0.05）')
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device
    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    g1_name = os.path.splitext(os.path.basename(args.test_file1))[0].capitalize()
    g2_name = os.path.splitext(os.path.basename(args.test_file2))[0].capitalize()
    
    output_dir = create_output_directory(args.vector_path, g1_name, g2_name, 
                                        base_dir="./projection_results")
    
    print(f"\n{'='*70}")
    print(f"{g1_name} vs {g2_name} 概念向量投影测试（Wasserstein + 方向判定）")
    print(f"模型: {args.model_path}")
    print(f"向量目录: {args.vector_path}")
    print(f"数据1: {args.test_file1} ({g1_name})")
    print(f"数据2: {args.test_file2} ({g2_name})")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print(f"判定: W-D>={args.wass_d_min}, p<={args.wass_p_max}")
    print(f"⭐ 方向: {g1_name} 应在 {g2_name} 的正方向（右侧）")
    print(f"{'='*70}\n")
    
    g1_texts = load_json_scenarios(args.test_file1)
    g2_texts = load_json_scenarios(args.test_file2)
    print(f"  ✅ {g1_name} 条数: {len(g1_texts)}")
    print(f"  ✅ {g2_name} 条数: {len(g2_texts)}")
    
    tester = ProjectionTester(args.model_path, args.vector_path, device)
    
    all_results: Dict[int, Dict[str, Any]] = {}
    for layer_idx in range(32):
        print(f"\n{'='*70}")
        print(f"测试第 {layer_idx} 层")
        print(f"{'='*70}")
        
        try:
            concept_vec, vecfile = tester.load_concept_vector(layer_idx)
        except FileNotFoundError as e:
            print(f"  ⚠️ {e}")
            continue
        
        g1_proj = tester.compute_projections(g1_texts, layer_idx, concept_vec)
        g2_proj = tester.compute_projections(g2_texts, layer_idx, concept_vec)
        
        stats = calculate_statistics(g1_proj, g2_proj)
        all_results[layer_idx] = {'statistics': stats}
        
        # 打印方向信息
        dir_info = stats['direction']
        dir_symbol = "✅" if dir_info['correct_direction'] else "❌"
        print(f"\n  结果:")
        print(f"    W-D = {stats['wass']['D']:.3f}, p = {stats['wass']['p']:.1e}")
        print(f"    {g1_name} median = {stats['group1']['median']:.4f}")
        print(f"    {g2_name} median = {stats['group2']['median']:.4f}")
        print(f"    方向判定: {dir_symbol} {'正确' if dir_info['correct_direction'] else '错误'}")
        
        vis_path = os.path.join(output_dir, "visualizations", f"layer_{layer_idx}_detail.png")
        visualize_single_layer_empirical(g1_proj, g2_proj, layer_idx, vis_path, 
                                        g1_name, g2_name, stats['wass'], stats['direction'])
        
        np.save(os.path.join(output_dir, "projections", f"layer_{layer_idx}_{g1_name}.npy"), g1_proj)
        np.save(os.path.join(output_dir, "projections", f"layer_{layer_idx}_{g2_name}.npy"), g2_proj)
    
    if len(all_results) > 0:
        print(f"\n{'='*70}")
        print("生成汇总可视化")
        print(f"{'='*70}")
        summary_path = os.path.join(output_dir, "visualizations", "all_layers_summary.png")
        visualize_all_layers_summary(all_results, summary_path, g1_name, g2_name)
    
    stats_data = {}
    for layer_idx, result in all_results.items():
        stats_data[f"layer_{layer_idx}"] = result['statistics']
    
    stats_path = os.path.join(output_dir, "statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 统计数据保存: {stats_path}")
    
    print(f"\n{'='*70}")
    print("生成测试报告")
    print(f"{'='*70}")
    generate_report(all_results, output_dir, g1_name, g2_name, args)
    
    print(f"\n{'='*70}")
    print("✅ 所有测试完成！")
    print(f"{'='*70}")
    print(f"\n所有结果保存在: {output_dir}/")
    print(f"  - 测试报告: test_report.txt (⭐ 包含方向判定)")
    print(f"  - 统计数据: statistics.json")
    print(f"  - 汇总图: visualizations/all_layers_summary.png (⭐ 方向标记)")
    print(f"  - 详细图: visualizations/layer_*_detail.png (⭐ 方向状态)")
    print(f"  - 原始数据: projections/")

if __name__ == "__main__":
    main()