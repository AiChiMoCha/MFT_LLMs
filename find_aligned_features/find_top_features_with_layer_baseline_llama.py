'''
This file finds the top-10 most-activated SAE features across all 5 moral foundations,
AND computes a per-layer random baseline for comparison.

The random baseline generates random unit vectors (same dimensionality as concept vectors)
and computes their cosine similarity with SAE decoder features FOR EACH LAYER.
This properly controls for layer-specific SAE decoder properties.
'''

import torch
import numpy as np
from sae_lens import SAE
import pandas as pd
import gc
import os

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [4, 8, 12, 16, 20, 24, 28] 
FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "sanctity"]
TOP_K = 10
N_RANDOM_SAMPLES = 100  # Number of random vectors to average over for stable baseline

def get_cosine_similarities(layer, foundation, device):
    """
    Calculates cosine similarity between a specific MFT foundation vector 
    and all SAE features for a given layer.
    """
    sae = SAE.from_pretrained(
        release="llama-3.1-8b-instruct-andyrdt", 
        sae_id=f"resid_post_layer_{layer-1}_trainer_1", 
        device=device
    )
    
    file_path = (
        f"/project2/shrikann_35/bowenyi/mft/MFT_LLMs/MFV130/"
        f"llama-3.1-8b-instruct_{foundation}_vs_social_norms_enhanced_concept_vector/"
        f"concept_vectors/vectors_npy/"
        f"llama-3.1-8b-instruct_{foundation}_vs_social_norms_model_layers_{layer-1}_residual_stream_vector.npy"
    )
    
    if not os.path.exists(file_path):
        print(f"Warning: Vector file not found for {foundation} at layer {layer}")
        return None, None

    pv_np = np.load(file_path)
    persona_vec = torch.from_numpy(pv_np).float().to(device)

    if persona_vec.dim() == 2 and persona_vec.size(0) == 1:
        persona_vec = persona_vec.squeeze(0)
        
    W_dec = sae.W_dec
    v_hat = persona_vec / (persona_vec.norm() + 1e-8)
    feat_norms = W_dec.norm(dim=1, keepdim=True) + 1e-8
    F_norm = W_dec / feat_norms
    cos_sims = F_norm @ v_hat
    top_vals, top_inds = torch.topk(cos_sims, k=TOP_K)
    
    del sae, W_dec, F_norm, persona_vec
    torch.cuda.empty_cache()
    gc.collect()
    
    return top_vals.detach().cpu().numpy(), top_inds.detach().cpu().numpy()


def get_random_baseline_per_layer(layer, device, n_samples=N_RANDOM_SAMPLES):
    """
    Computes baseline cosine similarities using random unit vectors FOR A SPECIFIC LAYER.
    
    For each random vector, we compute its top-k cosine similarities with that layer's
    SAE decoder features, then average across all random samples.
    
    Returns the average top-k similarities and their standard deviations.
    """
    sae = SAE.from_pretrained(
        release="llama-3.1-8b-instruct-andyrdt", 
        sae_id=f"resid_post_layer_{layer-1}_trainer_1", 
        device=device
    )
    
    W_dec = sae.W_dec
    d_model = W_dec.shape[1]
    n_features = W_dec.shape[0]
    
    print(f"  Layer {layer}: {n_features} features, {d_model} dimensions")
    
    # Normalize SAE decoder features once
    feat_norms = W_dec.norm(dim=1, keepdim=True) + 1e-8
    F_norm = W_dec / feat_norms
    
    all_top_vals = []
    
    for i in range(n_samples):
        # Generate random unit vector
        random_vec = torch.randn(d_model, device=device)
        random_vec = random_vec / (random_vec.norm() + 1e-8)
        
        # Compute similarities
        cos_sims = F_norm @ random_vec
        top_vals, _ = torch.topk(cos_sims, k=TOP_K)
        all_top_vals.append(top_vals.detach().cpu().numpy())
    
    # Stack and compute statistics
    all_top_vals = np.array(all_top_vals)  # Shape: [n_samples, TOP_K]
    avg_top_vals = all_top_vals.mean(axis=0)  # Shape: [TOP_K]
    std_top_vals = all_top_vals.std(axis=0)   # Shape: [TOP_K]
    
    del sae, W_dec, F_norm
    torch.cuda.empty_cache()
    gc.collect()
    
    return avg_top_vals, std_top_vals


# --- Main Execution Loop ---
results = []
baseline_results = []

print(f"Starting scan across {len(LAYERS)} layers for {len(FOUNDATIONS)} foundations...")

# Compute foundation results
for foundation in FOUNDATIONS:
    for layer in LAYERS:
        print(f"Processing: {foundation.capitalize()} - Layer {layer}")
        try:
            scores, indices = get_cosine_similarities(layer, foundation, DEVICE)
            
            if scores is not None:
                for rank, (score, idx) in enumerate(zip(scores, indices)):
                    results.append({
                        "Foundation": foundation,
                        "Layer": layer,
                        "Feature_ID": idx,
                        "Cosine_Similarity": score,
                        "Rank_in_Layer": rank + 1
                    })
        except Exception as e:
            print(f"Error: {e}")
            print(f"Empty files at {foundation} - layer {layer}")
            continue

# Compute random baseline PER LAYER
print("\n" + "="*50)
print("Computing per-layer random baseline...")
print("="*50)

for layer in LAYERS:
    print(f"Baseline - Layer {layer}")
    try:
        avg_vals, std_vals = get_random_baseline_per_layer(layer, DEVICE)
        for rank, (avg_score, std_score) in enumerate(zip(avg_vals, std_vals)):
            baseline_results.append({
                "Foundation": "random_baseline",
                "Layer": layer,
                "Feature_ID": -1,
                "Cosine_Similarity": avg_score,
                "Cosine_Similarity_Std": std_score,
                "Rank_in_Layer": rank + 1
            })
    except Exception as e:
        print(f"Error computing baseline at layer {layer}: {e}")
        continue

# --- Save Results ---
df = pd.DataFrame(results)
df_baseline = pd.DataFrame(baseline_results)

df_sorted = df.sort_values(by=["Foundation", "Cosine_Similarity"], ascending=[True, False])

print("\nTop Global Matches per Foundation:")
print(df_sorted.groupby("Foundation").head(3))

# Print baseline summary
print("\n" + "="*50)
print("Per-Layer Baseline Summary (Top-3 Average):")
print("="*50)
baseline_top3 = df_baseline[df_baseline["Rank_in_Layer"] <= 3].groupby("Layer")["Cosine_Similarity"].mean()
print(baseline_top3)

df_sorted.to_csv("mft_sae_feature_candidates.csv", index=False)
df_baseline.to_csv("mft_sae_random_baseline.csv", index=False)

print("\nSaved candidates to 'mft_sae_feature_candidates.csv'")
print("Saved baseline to 'mft_sae_random_baseline.csv'")
