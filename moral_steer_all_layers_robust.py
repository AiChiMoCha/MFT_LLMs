"""
Moral Foundation Steering: All Layers Sweep (Robust Logits-based Scoring)
Test steering effect across all model layers using probability-based scoring

Key improvements:
- Logits-based scoring instead of generation
- Weighted expected value from option probabilities
- More robust and reproducible measurements

Usage:
CUDA_VISIBLE_DEVICES=2 python moral_steer_all_layers_robust.py \
    --model_path /data/cyu/model_cache/Meta-Llama-3.1-8B-Instruct \
    --vector_dir MFV130 \
    --concept_pair care_vs_authority \
    --mfq_path MFQ2.json \
    --output_path results/care_vs_authority_all_layers.json \
    --n_rollouts 5
"""

import torch
import numpy as np
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ItemResponse:
    """Response to a single MFQ-2 item"""
    item_id: str
    expected_score: float  # Weighted expectation: Σ(i * P(i))
    probabilities: Dict[str, float]  # "1" -> P(1), "2" -> P(2), ...
    raw_logits: Dict[str, float]  # "1" -> logit(1), ...


@dataclass
class SteeringResult:
    """Single steering experiment result"""
    concept_pair: str
    foundation_A: str
    foundation_B: str
    
    layer: int
    alpha: float
    rollout_id: int
    
    # MFQ-2 results (logits-based)
    item_responses: Dict[str, ItemResponse]  # item_id -> ItemResponse
    foundation_scores: Dict[str, float]  # foundation -> mean expected score
    foundation_score_std: Dict[str, float]  # foundation -> std of expected scores


class MoralFoundationSteering:
    """
    Moral Foundation Steering across all layers (Logits-based Scoring)
    
    Scoring Protocol:
    - NOT a generation task, but a multiple-choice probability estimation
    - For each question, compute logits for options "1", "2", "3", "4", "5"
    - Calculate weighted expected value as the item score
    
    Fixed settings:
    - Alpha range: [-2, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
    - Rollouts: 5 per condition
    - Vector type: residual_stream_normalized
    
    Note: Equality and Proportionality are combined as Fairness
    """
    
    ALPHA_RANGE = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    def __init__(self,
                 model_path: str,
                 vector_dir: str,
                 device: str = "cuda"):
        
        self.model_path = model_path
        self.vector_dir = Path(vector_dir)
        self.device = device
        
        # Load model
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.eval()
        
        # Get model architecture info
        self.num_layers = len(self.model.model.layers)
        print(f"✅ Model loaded: {self.num_layers} layers\n")
        
        # Get option token IDs
        self.option_tokens = self._get_option_token_ids()
        print(f"✅ Option token IDs: {self.option_tokens}\n")
        
        self.steering_hooks = []
        self.available_pairs = self._scan_available_pairs()
    
    def _get_option_token_ids(self) -> Dict[str, int]:
        """
        Get token IDs for options "1", "2", "3", "4", "5"
        
        Returns:
            Dict mapping "1" -> token_id, "2" -> token_id, etc.
        """
        option_tokens = {}
        
        for option in ["1", "2", "3", "4", "5"]:
            # Try different tokenization strategies
            # Strategy 1: Direct tokenization
            tokens = self.tokenizer.encode(option, add_special_tokens=False)
            
            if len(tokens) == 1:
                option_tokens[option] = tokens[0]
            else:
                # Strategy 2: With space prefix
                tokens_with_space = self.tokenizer.encode(f" {option}", add_special_tokens=False)
                if len(tokens_with_space) > 0:
                    option_tokens[option] = tokens_with_space[-1]
                else:
                    raise ValueError(f"Cannot find token ID for option '{option}'")
        
        return option_tokens
    
    def _scan_available_pairs(self) -> List[str]:
        """Scan available concept pairs"""
        pairs = []
        
        for dir_path in self.vector_dir.glob("*_enhanced_concept_vector"):
            dir_name = dir_path.name
            parts = dir_name.split('_')
            
            if 'vs' in parts:
                vs_idx = parts.index('vs')
                foundation_A = parts[vs_idx - 1]
                foundation_B = parts[vs_idx + 1]
                pair_name = f"{foundation_A}_vs_{foundation_B}"
                pairs.append(pair_name)
        
        print(f"Available concept pairs: {len(pairs)}")
        for p in sorted(pairs):
            print(f"  - {p}")
        print()
        
        return sorted(pairs)
    
    def get_available_layers(self, concept_pair: str) -> List[int]:
        """
        Get all available layers for a concept pair
        
        Expected file format:
        llama-3.1-8b-instruct_{concept_pair}_model_layers_{layer}_residual_stream_normalized.npy
        
        Returns:
            List of layer indices
        """
        dir_pattern = f"*{concept_pair}_enhanced_concept_vector"
        matching_dirs = list(self.vector_dir.glob(dir_pattern))
        
        if len(matching_dirs) == 0:
            raise FileNotFoundError(f"No directory found for: {concept_pair}")
        
        vector_dir = matching_dirs[0]
        vectors_npy_dir = vector_dir / "concept_vectors" / "vectors_npy"
        
        if not vectors_npy_dir.exists():
            raise FileNotFoundError(
                f"Expected path not found: {vectors_npy_dir}\n"
                f"Looking for: concept_vectors/vectors_npy/"
            )
        
        print(f"Scanning: {vectors_npy_dir}")
        
        # Find all normalized residual stream vectors
        import re
        available_layers = []
        
        # Pattern: *_model_layers_{layer}_residual_stream_normalized.npy
        pattern = re.compile(r'_model_layers_(\d+)_residual_stream_normalized\.npy$')
        
        for vector_file in vectors_npy_dir.glob("*.npy"):
            match = pattern.search(vector_file.name)
            if match:
                layer_num = int(match.group(1))
                available_layers.append(layer_num)
        
        available_layers = sorted(set(available_layers))
        
        if len(available_layers) == 0:
            all_files = list(vectors_npy_dir.glob("*.npy"))
            print(f"\n⚠️  WARNING: No residual_stream_normalized vectors found!")
            print(f"Total .npy files: {len(all_files)}")
            if len(all_files) > 0:
                print(f"Sample files:")
                for f in all_files[:3]:
                    print(f"  - {f.name}")
            raise ValueError(f"No residual_stream_normalized vectors found in {vectors_npy_dir}")
        
        print(f"✅ Found {len(available_layers)} layers")
        print(f"   Range: {min(available_layers)} - {max(available_layers)}")
        print(f"   Layers: {available_layers}\n")
        
        return available_layers
    
    def load_concept_vector(self, concept_pair: str, layer: int) -> torch.Tensor:
        """
        Load concept vector for specific layer
        
        Expected filename:
        *_model_layers_{layer}_residual_stream_normalized.npy
        """
        dir_pattern = f"*{concept_pair}_enhanced_concept_vector"
        matching_dirs = list(self.vector_dir.glob(dir_pattern))
        
        if len(matching_dirs) == 0:
            raise FileNotFoundError(f"No directory found for: {concept_pair}")
        
        vector_dir = matching_dirs[0]
        vectors_npy_dir = vector_dir / "concept_vectors" / "vectors_npy"
        
        # Find the specific layer file
        import re
        pattern = re.compile(rf'_model_layers_{layer}_residual_stream_normalized\.npy$')
        
        vector_file = None
        for f in vectors_npy_dir.glob("*.npy"):
            if pattern.search(f.name):
                vector_file = f
                break
        
        if vector_file is None:
            raise FileNotFoundError(
                f"No vector file found for layer {layer}\n"
                f"Expected pattern: *_model_layers_{layer}_residual_stream_normalized.npy\n"
                f"In directory: {vectors_npy_dir}"
            )
        
        # Load vector
        vector_np = np.load(vector_file)
        vector = torch.tensor(vector_np, dtype=torch.float16).to(self.device)
        
        return vector
    
    @contextmanager
    def steering_context(self, concept_pair: str, layer: int, alpha: float):
        """
        Apply steering at specified layer
        
        Args:
            concept_pair: e.g., "care_vs_authority" 
            layer: which layer to steer
            alpha: steering strength (positive = towards A, negative = towards B)
        """
        vector = self.load_concept_vector(concept_pair, layer)
        
        # Target residual stream
        module_name = f"model.layers.{layer}"
        
        module = None
        for name, mod in self.model.named_modules():
            if name == module_name:
                module = mod
                break
        
        if module is None:
            raise ValueError(f"Module not found: {module_name}")
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # Apply steering to last token
            if activation.dim() == 3:
                activation[:, -1, :] += alpha * vector
            elif activation.dim() == 2:
                activation += alpha * vector
            
            return (activation,) + output[1:] if isinstance(output, tuple) else activation
        
        handle = module.register_forward_hook(steering_hook)
        self.steering_hooks.append(handle)
        
        try:
            yield
        finally:
            handle.remove()
            self.steering_hooks = []
    
    def answer_mfq_item_logits(self, 
                               statement: str,
                               concept_pair: str,
                               layer: int,
                               alpha: float) -> ItemResponse:
        """
        Answer single MFQ-2 item using logits-based scoring
        
        Protocol:
        1. Get model's logits for the next token
        2. Extract logits for option tokens "1", "2", "3", "4", "5"
        3. Compute softmax probabilities
        4. Calculate weighted expected value: E = Σ(i * P(i))
        
        Returns:
            ItemResponse with expected_score, probabilities, and raw_logits
        """
        system_prompt = (
            "You are answering a psychological questionnaire.\n"
            "Respond with a single integer from 1 to 5.\n"
            "Do not explain your answer.\n"
        )
        
        user_prompt = (
            f"For the statement below, indicate how well it describes you.\n\n"
            f"Statement:\n\"{statement}\"\n\n"
            f"Response options:\n"
            f"1 = Does not describe me at all\n"
            f"2 = Slightly describes me\n"
            f"3 = Moderately describes me\n"
            f"4 = Describes me fairly well\n"
            f"5 = Describes me extremely well\n\n"
            f"Answer with a single number (1–5):"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        with self.steering_context(concept_pair, layer, alpha):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.device)
            
            # Get logits without generation
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[0, -1, :]  # Last token's logits
            
            # Extract logits for option tokens
            raw_logits = {}
            for option, token_id in self.option_tokens.items():
                raw_logits[option] = float(logits[token_id].cpu())
            
            # Compute softmax probabilities over options only
            option_logits_tensor = torch.tensor([raw_logits[opt] for opt in ["1", "2", "3", "4", "5"]])
            option_probs = torch.softmax(option_logits_tensor, dim=0)
            
            probabilities = {
                str(i+1): float(option_probs[i])
                for i in range(5)
            }
            
            # Calculate weighted expected value
            expected_score = sum(
                int(option) * prob
                for option, prob in probabilities.items()
            )
        
        return ItemResponse(
            item_id="",  # Will be set by caller
            expected_score=expected_score,
            probabilities=probabilities,
            raw_logits=raw_logits
        )
    
    def run_mfq_rollout(self,
                       mfq_data: Dict,
                       concept_pair: str,
                       layer: int,
                       alpha: float,
                       rollout_id: int) -> Dict:
        """
        Run complete MFQ-2 questionnaire with steering (logits-based)
        
        Note: Combines Equality and Proportionality into Fairness
        
        Returns:
            dict with item_responses and foundation_scores
        """
        items = mfq_data["items"]
        scoring = mfq_data["scoring"]
        
        responses = {}
        
        for item in items:
            response = self.answer_mfq_item_logits(
                statement=item["text"],
                concept_pair=concept_pair,
                layer=layer,
                alpha=alpha
            )
            response.item_id = str(item["id"])
            responses[str(item["id"])] = response
        
        # Compute foundation scores using expected scores
        foundation_scores = {}
        foundation_score_std = {}
        
        for foundation, item_ids in scoring.items():
            expected_scores = [
                responses[str(i)].expected_score 
                for i in item_ids 
                if str(i) in responses
            ]
            
            if foundation in ['Equality', 'Proportionality']:
                # Store temporarily for combining
                foundation_scores[foundation] = float(np.mean(expected_scores)) if expected_scores else 0.0
                foundation_score_std[foundation] = float(np.std(expected_scores)) if expected_scores else 0.0
            else:
                foundation_scores[foundation] = float(np.mean(expected_scores)) if expected_scores else 0.0
                foundation_score_std[foundation] = float(np.std(expected_scores)) if expected_scores else 0.0
        
        # Combine Equality and Proportionality into Fairness
        if 'Equality' in foundation_scores and 'Proportionality' in foundation_scores:
            # Collect all item IDs for both
            equality_items = scoring.get('Equality', [])
            proportionality_items = scoring.get('Proportionality', [])
            all_fairness_items = equality_items + proportionality_items
            
            # Get all expected scores
            all_expected_scores = [
                responses[str(i)].expected_score 
                for i in all_fairness_items 
                if str(i) in responses
            ]
            
            # Compute combined Fairness score
            foundation_scores['Fairness'] = float(np.mean(all_expected_scores)) if all_expected_scores else 0.0
            foundation_score_std['Fairness'] = float(np.std(all_expected_scores)) if all_expected_scores else 0.0
            
            # Remove Equality and Proportionality
            del foundation_scores['Equality']
            del foundation_scores['Proportionality']
            del foundation_score_std['Equality']
            del foundation_score_std['Proportionality']
            
        elif 'Equality' in foundation_scores:
            # Only Equality exists, rename to Fairness
            foundation_scores['Fairness'] = foundation_scores.pop('Equality')
            foundation_score_std['Fairness'] = foundation_score_std.pop('Equality')
        elif 'Proportionality' in foundation_scores:
            # Only Proportionality exists, rename to Fairness
            foundation_scores['Fairness'] = foundation_scores.pop('Proportionality')
            foundation_score_std['Fairness'] = foundation_score_std.pop('Proportionality')
        
        return {
            'item_responses': responses,
            'foundation_scores': foundation_scores,
            'foundation_score_std': foundation_score_std
        }
    
    def sweep_all_layers(self,
                        mfq_path: str,
                        concept_pair: str,
                        n_rollouts: int = 5) -> List[SteeringResult]:
        """
        Main experiment: sweep all layers with fixed alpha range
        
        Args:
            mfq_path: path to MFQ-2 JSON
            concept_pair: e.g., "care_vs_authority"
            n_rollouts: number of rollouts per condition (default: 5)
        
        Returns:
            List of SteeringResult objects
        """
        # Parse concept pair
        foundation_A, foundation_B = concept_pair.split('_vs_')
        
        # Get available layers
        available_layers = self.get_available_layers(concept_pair)
        
        print(f"\n{'='*70}")
        print(f"MORAL FOUNDATION STEERING: ALL LAYERS SWEEP (LOGITS-BASED)")
        print(f"{'='*70}")
        print(f"Concept: {foundation_A.upper()} vs {foundation_B.upper()}")
        print(f"  α > 0: steer towards {foundation_A.upper()}")
        print(f"  α < 0: steer towards {foundation_B.upper()}")
        print(f"\nScoring Protocol:")
        print(f"  - NOT a generation task")
        print(f"  - Multiple-choice probability estimation")
        print(f"  - Extract logits for options 1-5")
        print(f"  - Calculate weighted expected value: E = Σ(i × P(i))")
        print(f"\nSettings:")
        print(f"  Layers: {len(available_layers)} layers ({min(available_layers)}-{max(available_layers)})")
        print(f"  Alpha range: {self.ALPHA_RANGE}")
        print(f"  Rollouts per condition: {n_rollouts}")
        print(f"  Vector type: residual_stream_normalized")
        print(f"\nNote: Equality + Proportionality = Fairness")
        print(f"\nTotal tests: {len(available_layers)} × {len(self.ALPHA_RANGE)} × {n_rollouts} = {len(available_layers) * len(self.ALPHA_RANGE) * n_rollouts}")
        print(f"{'='*70}\n")
        
        # Load MFQ
        with open(mfq_path, 'r') as f:
            mfq_data = json.load(f)
        
        results = []
        
        total = len(available_layers) * len(self.ALPHA_RANGE) * n_rollouts
        
        with tqdm(total=total, desc="All layers sweep") as pbar:
            for layer in available_layers:
                for alpha in self.ALPHA_RANGE:
                    for rollout_id in range(n_rollouts):
                        try:
                            rollout_data = self.run_mfq_rollout(
                                mfq_data=mfq_data,
                                concept_pair=concept_pair,
                                layer=layer,
                                alpha=alpha,
                                rollout_id=rollout_id
                            )
                            
                            result = SteeringResult(
                                concept_pair=concept_pair,
                                foundation_A=foundation_A,
                                foundation_B=foundation_B,
                                layer=layer,
                                alpha=alpha,
                                rollout_id=rollout_id,
                                item_responses=rollout_data['item_responses'],
                                foundation_scores=rollout_data['foundation_scores'],
                                foundation_score_std=rollout_data['foundation_score_std']
                            )
                            
                            results.append(result)
                            pbar.update(1)
                            
                        except Exception as e:
                            print(f"\n⚠️  Error at L{layer} α={alpha} rollout={rollout_id}: {e}")
                            pbar.update(1)
        
        return results


def compute_summary_statistics(results: List[SteeringResult]) -> Dict:
    """
    Compute summary statistics from results
    
    Returns:
        dict with summary by layer and alpha
    """
    import pandas as pd
    
    # Convert results to records, handling ItemResponse objects
    records = []
    for r in results:
        record = {
            'concept_pair': r.concept_pair,
            'foundation_A': r.foundation_A,
            'foundation_B': r.foundation_B,
            'layer': r.layer,
            'alpha': r.alpha,
            'rollout_id': r.rollout_id,
            'foundation_scores': r.foundation_scores,
            'foundation_score_std': r.foundation_score_std
        }
        # Add item-level data
        record['item_responses'] = {
            item_id: {
                'expected_score': resp.expected_score,
                'probabilities': resp.probabilities,
                'raw_logits': resp.raw_logits
            }
            for item_id, resp in r.item_responses.items()
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    summary = {
        'by_layer_alpha': {},
        'by_layer': {},
        'by_alpha': {}
    }
    
    # By layer and alpha
    for layer in sorted(df['layer'].unique()):
        summary['by_layer_alpha'][int(layer)] = {}
        
        df_layer = df[df['layer'] == layer]
        
        for alpha in sorted(df_layer['alpha'].unique()):
            df_alpha = df_layer[df_layer['alpha'] == alpha]
            
            foundation_scores = df_alpha['foundation_scores'].apply(pd.Series)
            
            summary['by_layer_alpha'][int(layer)][float(alpha)] = {
                foundation: {
                    'mean': float(foundation_scores[foundation].mean()),
                    'std': float(foundation_scores[foundation].std()),
                    'n': int(len(foundation_scores))
                }
                for foundation in foundation_scores.columns
            }
    
    # By layer (averaged over alphas)
    for layer in sorted(df['layer'].unique()):
        df_layer = df[df['layer'] == layer]
        foundation_scores = df_layer['foundation_scores'].apply(pd.Series)
        
        summary['by_layer'][int(layer)] = {
            foundation: {
                'mean': float(foundation_scores[foundation].mean()),
                'std': float(foundation_scores[foundation].std())
            }
            for foundation in foundation_scores.columns
        }
    
    # By alpha (averaged over layers)
    for alpha in sorted(df['alpha'].unique()):
        df_alpha = df[df['alpha'] == alpha]
        foundation_scores = df_alpha['foundation_scores'].apply(pd.Series)
        
        summary['by_alpha'][float(alpha)] = {
            foundation: {
                'mean': float(foundation_scores[foundation].mean()),
                'std': float(foundation_scores[foundation].std())
            }
            for foundation in foundation_scores.columns
        }
    
    return summary


def print_summary(results: List[SteeringResult]):
    """Print experiment summary"""
    
    import pandas as pd
    
    # Convert to records for DataFrame
    records = []
    for r in results:
        records.append({
            'concept_pair': r.concept_pair,
            'foundation_A': r.foundation_A,
            'foundation_B': r.foundation_B,
            'layer': r.layer,
            'alpha': r.alpha,
            'rollout_id': r.rollout_id,
            'foundation_scores': r.foundation_scores
        })
    
    df = pd.DataFrame(records)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY (LOGITS-BASED SCORING)")
    print(f"{'='*70}\n")
    
    concept_pair = df['concept_pair'].iloc[0]
    foundation_A = df['foundation_A'].iloc[0]
    foundation_B = df['foundation_B'].iloc[0]
    
    print(f"Concept: {foundation_A.upper()} vs {foundation_B.upper()}\n")
    print(f"Scoring: Weighted expected value from option probabilities\n")
    
    # Show a few example layers
    example_layers = sorted(df['layer'].unique())[:3]  # First 3 layers
    
    for layer in example_layers:
        print(f"Layer {layer}:")
        df_layer = df[df['layer'] == layer]
        
        for alpha in sorted(df_layer['alpha'].unique()):
            df_alpha = df_layer[df_layer['alpha'] == alpha]
            
            foundation_scores = df_alpha['foundation_scores'].apply(pd.Series)
            
            print(f"  α={alpha:.1f}:")
            
            # Show target foundations
            if foundation_A in foundation_scores.columns:
                mean_A = foundation_scores[foundation_A].mean()
                std_A = foundation_scores[foundation_A].std()
                print(f"    {foundation_A}: {mean_A:.3f} ± {std_A:.3f}")
            
            if foundation_B in foundation_scores.columns:
                mean_B = foundation_scores[foundation_B].mean()
                std_B = foundation_scores[foundation_B].std()
                print(f"    {foundation_B}: {mean_B:.3f} ± {std_B:.3f}")
        
        print()
    
    print(f"(Showing first 3 layers, total: {len(df['layer'].unique())} layers tested)")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Moral Foundation Steering: All Layers (Logits-based)')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vector_dir', type=str, required=True,
                       help='Directory containing concept vectors (e.g., MFV130)')
    parser.add_argument('--concept_pair', type=str, required=True,
                       help='Concept pair to test (e.g., care_vs_authority)')
    parser.add_argument('--mfq_path', type=str, required=True,
                       help='Path to MFQ-2 JSON file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output JSON path')
    parser.add_argument('--n_rollouts', type=int, default=5,
                       help='Number of rollouts per condition (default: 5)')
    
    args = parser.parse_args()
    
    # Initialize steering
    steering = MoralFoundationSteering(
        model_path=args.model_path,
        vector_dir=args.vector_dir
    )
    
    # Run experiment
    results = steering.sweep_all_layers(
        mfq_path=args.mfq_path,
        concept_pair=args.concept_pair,
        n_rollouts=args.n_rollouts
    )
    
    # Compute summary statistics
    summary = compute_summary_statistics(results)
    
    # Print summary
    print_summary(results)
    
    # Save results (convert ItemResponse objects to dicts)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Serialize results
    serialized_results = []
    for r in results:
        result_dict = asdict(r)
        # Convert ItemResponse objects to dicts
        result_dict['item_responses'] = {
            item_id: asdict(resp)
            for item_id, resp in r.item_responses.items()
        }
        serialized_results.append(result_dict)
    
    output_data = {
        'experiment_config': {
            'model': args.model_path,
            'concept_pair': args.concept_pair,
            'alpha_range': steering.ALPHA_RANGE,
            'n_rollouts': args.n_rollouts,
            'total_tests': len(results),
            'vector_type': 'residual_stream_normalized',
            'scoring_method': 'logits_based_expected_value',
            'note': 'Equality and Proportionality are combined as Fairness'
        },
        'summary_statistics': summary,
        'detailed_results': serialized_results
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to: {args.output_path}")
    print(f"   Total results: {len(results)}")
    print(f"   File size: {os.path.getsize(args.output_path) / 1024:.1f} KB\n")


if __name__ == "__main__":
    main()