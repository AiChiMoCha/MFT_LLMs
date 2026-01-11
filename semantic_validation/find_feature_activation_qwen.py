"""
Mine Top-Activating Text Examples from FineWeb for SAE Features

This script:
1. Loads a random sample from FineWeb dataset
2. For each candidate feature, computes SAE activations on the corpus
3. Extracts the top-40 max-activating text examples
4. Saves results to JSON

Requirements:
    pip install torch transformers datasets sae_lens pandas tqdm

Usage:
    python mine_fineweb_activations.py

Author: Bowen
"""
import heapq

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
import gc
import os
import time

# ============================================================
# CONFIGURATION - EDIT THESE AS NEEDED
# ============================================================

# Input/Output paths
CSV_PATH = "mft_sae_feature_candidates_qwen.csv"
N_CORPUS_SAMPLES = 50000
model = 'qwen'
OUTPUT_PATH = f"result/fineweb_feature_activations_qwen.json"
CORPUS_CACHE_PATH = f"fineweb_corpus_cache_a40_{N_CORPUS_SAMPLES}.json"


# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SAE_RELEASE = "qwen2.5-7b-instruct-andyrdt"  # SAELens release name

# Dataset configuration  
DATASET_NAME = "HuggingFaceFW/fineweb-edu"  # Using fineweb-edu for better quality

MIN_TEXT_LENGTH = 100  # Minimum characters
MAX_TEXT_LENGTH = 800  # Maximum characters (will truncate)

# Processing configuration
BATCH_SIZE = 16  # Adjust based on your GPU memory (8 is safer)
MAX_TOKENS = 512  # Max tokens per text for the model
TOP_K_EXAMPLES = 40  # Number of top-activating examples to keep
DEVICE = "cuda"  # or "cpu"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ActivationExample:
    """Represents a single activation example."""
    text: str
    max_activation: float
    max_activation_token_index: int
    mean_activation: float


@dataclass
class FeatureResult:
    """Results for a single feature."""
    foundation: str
    layer: int
    feature_id: int
    sae_layer: int  # layer - 1
    cosine_similarity: float
    rank_in_layer: int
    top_activations: List[Dict[str, Any]]


# ============================================================
# DATASET LOADING
# ============================================================

def load_fineweb_sample(
    dataset_name: str = DATASET_NAME,
    n_samples: int = N_CORPUS_SAMPLES,
    min_length: int = MIN_TEXT_LENGTH,
    max_length: int = MAX_TEXT_LENGTH,
    seed: int = 42,
    text_field: str = "text"
) -> List[str]:
    """
    Load a random sample of texts from FineWeb dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        n_samples: Number of texts to sample
        min_length: Minimum character length
        max_length: Maximum character length (will truncate)
        seed: Random seed for reproducibility
        text_field: Name of the text field in the dataset
        
    Returns:
        List of text strings
    """
    from datasets import load_dataset
    
    print(f"Loading {n_samples} samples from {dataset_name}...")
    print("This may take a few minutes for streaming...")
    
    # Use streaming to avoid downloading the entire dataset
    dataset = load_dataset(
        dataset_name,
        name="sample-10BT",  # Use the 10BT sample for faster loading
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    # Shuffle with seed for reproducibility
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    texts = []
    seen = 0
    
    for example in tqdm(dataset, desc="Sampling texts", total=n_samples * 2):
        text = example.get(text_field, '')
        seen += 1
        
        # Filter by length
        if len(text) >= min_length:
            # Truncate if too long
            texts.append(text[:max_length])
            
        if len(texts) >= n_samples:
            break
            
        # Safety check to avoid infinite loop
        if seen > n_samples * 10:
            print(f"Warning: Only found {len(texts)} valid texts after scanning {seen} examples")
            break
    
    print(f"Loaded {len(texts)} texts (scanned {seen} total)")
    return texts


def save_corpus_cache(texts: List[str], cache_path: str):
    """Save corpus to disk for reuse."""
    print(f"Caching corpus to {cache_path}...")
    with open(cache_path, 'w') as f:
        json.dump(texts, f)


def load_corpus_cache(cache_path: str) -> Optional[List[str]]:
    """Load cached corpus if available."""
    if os.path.exists(cache_path):
        print(f"Loading cached corpus from {cache_path}...")
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None


# ============================================================
# SAE ACTIVATION COMPUTATION
# ============================================================

def get_sae_id(layer: int) -> str:
    """
    Get the SAE ID for a given layer.
    
    Args:
        layer: Your layer number (1-indexed in your convention)
        
    Returns:
        SAE ID string for SAELens
    """
    sae_layer = layer - 1  # Convert to 0-indexed SAE layer
    # Format for qwen-3.1-8b-instruct-andyrdt release
    return f"resid_post_layer_{sae_layer}_trainer_1"


def compute_feature_activations_batched(
    texts: List[str],
    layer: int,
    feature_ids: List[int],
    model=None,
    tokenizer=None,
    sae=None,
    device: str = "cuda",
    batch_size: int = BATCH_SIZE,
    max_tokens: int = MAX_TOKENS,
    top_k: int = TOP_K_EXAMPLES,
):
    """
    Compute activations for multiple features in one pass through the corpus.

    Key optimizations:
      (1) Online top-k per feature using a min-heap: O(#features * top_k) memory.
      (2) Vectorized extraction over feature_ids: avoid Python loops over features for tensor reductions.

    Returns:
        heaps: Dict[feature_id] -> min-heap of (max_act, max_idx, mean_act, text)
        model, tokenizer, sae: so caller can reuse
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    # IMPORTANT: keep feature_ids order stable
    feature_ids = list(feature_ids)

    # Per-feature min-heaps storing top-k examples
    heaps: Dict[int, List[Tuple[float, int, float, str]]] = {fid: [] for fid in feature_ids}

    feature_ids_tensor = torch.tensor(feature_ids, device=device, dtype=torch.long)

    # Load model if not provided
    if model is None:
        print(f"Loading model {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            print("Using Flash Attention 2")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Using standard attention")
        model.eval()

    # Load SAE if not provided
    if sae is None:
        sae_id = get_sae_id(layer)
        print(f"Loading SAE {sae_id} from {SAE_RELEASE}...")
        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_id,
            device=device
        )[0]
        sae.eval()

    print(f"Processing {len(texts)} texts for {len(feature_ids)} features at layer {layer}...")

    total_batches = (len(texts) + batch_size - 1) // batch_size
    start_time = time.time()

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Layer {layer}", total=total_batches):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_states = outputs.hidden_states[layer]  # [batch, seq, d_model]
            sae_acts = sae.encode(hidden_states.float())  # [batch, seq, n_features]
            attention_mask = inputs.attention_mask        # [batch, seq]

            # Process each sequence in the batch (still necessary due to variable lengths)
            for j, text in enumerate(batch_texts):
                valid_length = int(attention_mask[j].sum().item())
                if valid_length <= 0:
                    continue

                # [valid_seq, n_features]
                text_acts = sae_acts[j, :valid_length, :]

                # Vectorized slice: [valid_seq, n_selected]
                sel = text_acts.index_select(dim=1, index=feature_ids_tensor)

                # Compute per-feature stats vectorized
                # max_vals: [n_selected], argmax_pos: [n_selected]
                max_vals, argmax_pos = sel.max(dim=0)
                mean_vals = sel.mean(dim=0)

                # Update per-feature heaps (Python loop only over #features, not tokens)
                max_vals_cpu = max_vals.detach().float().cpu().numpy()
                argmax_cpu = argmax_pos.detach().cpu().numpy()
                mean_cpu = mean_vals.detach().float().cpu().numpy()

                for k_idx, fid in enumerate(feature_ids):
                    max_act = float(max_vals_cpu[k_idx])
                    max_idx = int(argmax_cpu[k_idx])
                    mean_act = float(mean_cpu[k_idx])

                    heap = heaps[fid]
                    item = (max_act, max_idx, mean_act, text)

                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    else:
                        # Keep only top_k by max_act
                        if max_act > heap[0][0]:
                            heapq.heapreplace(heap, item)

        if batch_num % 100 == 0:
            torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            batches_done = batch_num
            batches_remaining = total_batches - batches_done
            time_per_batch = elapsed / max(1, batches_done)
            eta_minutes = (batches_remaining * time_per_batch) / 60
            print(f"  Batch {batch_num}/{total_batches} | Elapsed: {elapsed/60:.1f}min | ETA: {eta_minutes:.1f}min")

    layer_time = time.time() - start_time
    print(f"  Layer {layer} completed in {layer_time/60:.1f} minutes")

    return heaps, model, tokenizer, sae



def get_top_k_activations(
    heap: List[Tuple[float, int, float, str]],
    top_k: int = TOP_K_EXAMPLES
) -> List[Dict[str, Any]]:
    """
    Convert a per-feature heap into a sorted list of top-k ActivationExample dicts.
    Heap contains (max_act, max_idx, mean_act, text).
    """
    # Sort descending by max_act
    sorted_results = sorted(heap, key=lambda x: x[0], reverse=True)[:top_k]

    top_examples = []
    for max_act, max_idx, mean_act, text in sorted_results:
        example = ActivationExample(
            text=text,
            max_activation=max_act,
            max_activation_token_index=max_idx,
            mean_activation=mean_act
        )
        top_examples.append(asdict(example))

    return top_examples



# ============================================================
# MAIN PIPELINE
# ============================================================

def process_all_features(
    csv_path: str = CSV_PATH,
    output_path: str = OUTPUT_PATH,
    corpus_cache_path: str = CORPUS_CACHE_PATH,
    n_corpus_samples: int = N_CORPUS_SAMPLES,
    top_k: int = TOP_K_EXAMPLES,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Main processing function.
    
    Processes features layer by layer for efficiency (only need to load
    each SAE once).
    """
    import torch
    
    total_start_time = time.time()
    
    # Load candidate features
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    print(f"Loading candidate features from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} features")
    print(f"Unique layers: {sorted(df['Layer'].unique())}")
    
    # Get unique (layer, feature_id) pairs
    unique_features = df.groupby(['Layer', 'Feature_ID']).first().reset_index()
    print(f"Unique (layer, feature) pairs: {len(unique_features)}")
    
    # Load or create corpus
    corpus = load_corpus_cache(corpus_cache_path)
    if corpus is None:
        corpus = load_fineweb_sample(n_samples=n_corpus_samples)
        save_corpus_cache(corpus, corpus_cache_path)
    
    print(f"Corpus size: {len(corpus)} texts")
    
    # Group features by layer for efficient processing
    layer_features = df.groupby('Layer')['Feature_ID'].apply(set).to_dict()
    n_layers = len(layer_features)
    
    # Estimate total time
    total_batches_per_layer = (len(corpus) + batch_size - 1) // batch_size
    print(f"\n{'='*60}")
    print("PROCESSING ESTIMATE")
    print(f"{'='*60}")
    print(f"Layers to process: {n_layers}")
    print(f"Batches per layer: {total_batches_per_layer}")
    print(f"Total batches: {n_layers * total_batches_per_layer}")
    print(f"Estimated time: {n_layers * total_batches_per_layer * 0.1 / 60:.0f}-{n_layers * total_batches_per_layer * 0.3 / 60:.0f} minutes")
    print(f"{'='*60}\n")
    
    # Store activation results keyed by (layer, feature_id)
    all_activations = {}
    
    # Process layer by layer
    model, tokenizer, sae = None, None, None
    layer_times = []
    
    for layer_idx, layer in enumerate(sorted(layer_features.keys())):
        feature_ids = list(layer_features[layer])
        print(f"\n{'='*60}")
        print(f"LAYER {layer} ({layer_idx + 1}/{n_layers}) - {len(feature_ids)} features")
        print(f"{'='*60}")
        
        # Clean up previous SAE
        if sae is not None:
            del sae
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Compute activations for all features at this layer
            heaps, model, tokenizer, sae = compute_feature_activations_batched(
            texts=corpus,
            layer=layer,
            feature_ids=feature_ids,
            model=model,
            tokenizer=tokenizer,
            sae=None,
            device=device,
            batch_size=batch_size,
            top_k=top_k,  # pass through
            )
            
            for fid in feature_ids:
                all_activations[(layer, fid)] = heaps[fid]

            
            # Track time for estimates
            if layer_idx == 0:
                first_layer_time = time.time() - total_start_time
                estimated_total = first_layer_time * n_layers
                print(f"\n  Based on first layer, estimated total time: {estimated_total/60:.1f} minutes")
                
        except Exception as e:
            print(f"Error processing layer {layer}: {e}")
            import traceback
            traceback.print_exc()
            # Still continue with other layers
            for fid in feature_ids:
                all_activations[(layer, fid)] = []
    
    # Clean up
    print("\nCleaning up GPU memory...")
    del model, tokenizer, sae
    torch.cuda.empty_cache()
    gc.collect()
    
    # Build final results
    print("\n" + "="*60)
    print("BUILDING RESULTS")
    print("="*60)
    results_list = []
    
    for _, row in tqdm(df.iterrows(), desc="Building results", total=len(df)):
        layer = row['Layer']
        feature_id = row['Feature_ID']
        
        activation_data = all_activations.get((layer, feature_id), [])
        top_acts = get_top_k_activations(activation_data, top_k)
        
        result = FeatureResult(
            foundation=row['Foundation'],
            layer=layer,
            feature_id=feature_id,
            sae_layer=layer - 1,
            cosine_similarity=row['Cosine_Similarity'],
            rank_in_layer=row['Rank_in_Layer'],
            top_activations=top_acts
        )
        results_list.append(asdict(result))
    
    # Save output
    output = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "corpus_dataset": DATASET_NAME,
        "corpus_size": len(corpus),
        "top_k_per_feature": top_k,
        "total_features": len(results_list),
        "features": results_list
    }
    
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Features processed: {len(results_list)}")
    print(f"Output saved to: {output_path}")
    
    return output


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Run the feature activation mining pipeline."""
    print("\n" + "="*60)
    print("FineWeb SAE Feature Activation Mining")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input CSV:     {CSV_PATH}")
    print(f"  Output JSON:   {OUTPUT_PATH}")
    print(f"  Corpus cache:  {CORPUS_CACHE_PATH}")
    print(f"  Model:         {MODEL_NAME}")
    print(f"  SAE Release:   {SAE_RELEASE}")
    print(f"  Corpus size:   {N_CORPUS_SAMPLES}")
    print(f"  Top-k:         {TOP_K_EXAMPLES}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Device:        {DEVICE}")
    print()
    
    process_all_features()


if __name__ == "__main__":
    main()