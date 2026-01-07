"""
Moral Foundations Differential Vector Experiment - PCA Method
Uses pairwise difference + PCA to extract concept vectors

Key differences from mean difference method:
1. Random pairing between target and control samples
2. Multiple rollouts concatenated
3. Extract PC1 and PC2 from differential activations
"""

import json
import torch
import numpy as np
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import warnings
import argparse
from datetime import datetime
import csv
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


class DifferentialVectorExperiment:
    """Differential Vector Experiment - Using pairwise difference + PCA"""
    
    def __init__(self, 
                 target_foundation: str = "fairness",
                 control_foundation: str = "social_norms",
                 model_name: str = "mistral-7b-instruct", 
                 model_path: str = "/data/cyu/SecAlign/mistral-7b-instruct-v0.1", 
                 device: str = "cuda",
                 monitoring_mode: str = "comprehensive",
                 temperature: float = 0.7,
                 max_new_tokens: int = 10,
                 n_rollouts: int = 5,
                 random_seed: int = 42):
        
        self.target_foundation = target_foundation.lower()
        self.control_foundation = control_foundation.lower()
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.monitoring_mode = monitoring_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.n_rollouts = n_rollouts
        self.random_seed = random_seed
        self.model = None
        self.tokenizer = None
        self.activation_storage = {}
        self.hooks = []
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        self.model_config = None
        self.total_layers = None
        self.output_dir = self._setup_output_directory()
        self.target_layers = []
        
        self.foundation_mapping = {
            'care': ['Care (e)', 'Care (p, h)', 'Care (p, a)'],
            'fairness': ['Fairness'],
            'loyalty': ['Loyalty'], 
            'authority': ['Authority'],
            'sanctity': ['Sanctity'],
            'liberty': ['Liberty'],
            'unrelated': ['Unrelated'],
            'social_norms': ['Social Norms']
        }
        
        print(f"Target foundation: {self.target_foundation.title()}")
        print(f"Control foundation: {self.control_foundation.title()}")
        print(f"Model: {self.model_name}")
        print(f"Method: Pairwise difference + PCA")
        print(f"Rollouts: {self.n_rollouts}")
        print(f"Random seed: {self.random_seed}")
        
    def _setup_output_directory(self) -> str:
        """Setup output directory"""
        base_name = f"{self.model_name}_{self.target_foundation}_vs_{self.control_foundation}_differential_pca"
        output_dir = f"MFV130/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/detailed_logs", exist_ok=True)
        os.makedirs(f"{output_dir}/differential_vectors", exist_ok=True)
        return output_dir
        
    def _check_model_compatibility(self):
        """Check model compatibility"""
        try:
            self.model_config = AutoConfig.from_pretrained(self.model_path)
            self.total_layers = self.model_config.num_hidden_layers
            print(f"  Model type: {self.model_config.model_type}")
            print(f"  Hidden layers: {self.total_layers}")
            print(f"  Hidden dimension: {self.model_config.hidden_size}")
            return True
        except Exception as e:
            print(f"❌ Model check failed: {e}")
            return False
            
    def _configure_monitoring_layers(self) -> List[int]:
        """Configure monitoring layers"""
        total_layers = self.total_layers or 32
        
        mode_configs = {
            "light": [total_layers//4, total_layers//2, total_layers*3//4],
            "comprehensive": [max(1, total_layers//6), total_layers//3, total_layers//2, 
                             total_layers*2//3, total_layers*5//6, total_layers-2],
            "dense": list(range(2, total_layers, max(1, total_layers//10))),
            "full": list(range(total_layers))
        }
        
        layers = mode_configs.get(self.monitoring_mode, mode_configs["comprehensive"])
        layers = [l for l in layers if 0 <= l < total_layers]
        
        print(f"Target layers: {layers}")
        return layers
        
    def load_model(self):
        """Load model"""
        print(f"Loading {self.model_name}...")
        
        if not self._check_model_compatibility():
            raise ValueError("Model compatibility check failed!")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.target_layers = self._configure_monitoring_layers()
        print("Model loaded successfully!")
    
    def load_moral_scenarios(self, json_path: str, unrelated_path: str = "unrelated_questions.json") -> Dict:
        """Load moral scenario data"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if os.path.exists(unrelated_path):
            with open(unrelated_path, 'r', encoding='utf-8') as f:
                unrelated_data = json.load(f)
                data['scenarios'].extend(unrelated_data['scenarios'])
        return data
    
    def filter_scenarios_by_foundation(self, scenarios: List[Dict], foundation_key: str) -> List[Dict]:
        """Filter scenarios"""
        if foundation_key.lower() in ["social_norms", "unrelated"]:
            target_name = "Social Norms" if foundation_key.lower() == "social_norms" else "Unrelated"
            return [s for s in scenarios if s['foundation'] == target_name]
        
        if foundation_key.lower() in self.foundation_mapping:
            target_foundations = self.foundation_mapping[foundation_key.lower()]
            return [s for s in scenarios 
                   if any(s['foundation'].startswith(tf.split()[0]) if '(' in tf 
                         else s['foundation'] == tf 
                         for tf in target_foundations)]
        return [s for s in scenarios if s['foundation'] == foundation_key]
    
    def create_moral_prompt(self, scenario: str) -> str:
        """Create prompt"""
        return f"""Rate the moral wrongness of this behavior on a 0–4 scale (0 = not at all wrong; 4 = extremely wrong):

"{scenario}"

Rating:"""
    
    def setup_activation_hooks(self):
        """Setup activation hooks"""
        self.activation_storage.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        for name, module in self.model.named_modules():
            for layer_num in self.target_layers:
                if name == f"model.layers.{layer_num}":
                    def make_hook(layer_name):
                        def hook_fn(module, input, output):
                            activation = output[0] if isinstance(output, tuple) else output
                            if activation.dim() == 3:
                                activation = activation[:, -1, :]
                            self.activation_storage[layer_name] = activation.detach().cpu().float().numpy()
                        return hook_fn
                    
                    self.hooks.append(module.register_forward_hook(make_hook(name)))
                    break
    
    def generate_with_activations(self, prompt: str) -> Tuple[str, Dict]:
        """Generate and collect activations"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.activation_storage.clear()
        
        with torch.no_grad():
            self.model(**inputs, output_hidden_states=True, use_cache=False)
        
        encoding_activations = {k: v.copy() for k, v in self.activation_storage.items()}
        
        for hook in self.hooks:
            hook.remove()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        self.setup_activation_hooks()
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text, encoding_activations
    
    def run_experiment(self, scenarios_data: Dict, n_samples_per_group: int = 200) -> Dict:
        """Run experiment - collect activations"""
        print(f"\nStarting experiment: {self.target_foundation.title()} vs {self.control_foundation.title()}")
        
        # Filter scenarios
        target_scenarios = self.filter_scenarios_by_foundation(
            scenarios_data['scenarios'], self.target_foundation
        )[:n_samples_per_group]
        
        control_scenarios = self.filter_scenarios_by_foundation(
            scenarios_data['scenarios'], self.control_foundation
        )[:n_samples_per_group]
        
        print(f"{self.target_foundation.title()}: {len(target_scenarios)} scenarios")
        print(f"{self.control_foundation.title()}: {len(control_scenarios)} scenarios")
        
        self.setup_activation_hooks()
        
        all_rollouts_data = []
        
        for rollout_idx in range(self.n_rollouts):
            print(f"\nRollout {rollout_idx + 1}/{self.n_rollouts}")
            
            target_data = []
            control_data = []
            
            # Process target scenarios
            for i, scenario in enumerate(tqdm(target_scenarios, desc=self.target_foundation.title())):
                prompt = self.create_moral_prompt(scenario['scenario'])
                try:
                    _, activations = self.generate_with_activations(prompt)
                    target_data.append({'activations': activations})
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            # Process control scenarios
            for i, scenario in enumerate(tqdm(control_scenarios, desc=self.control_foundation.title())):
                prompt = self.create_moral_prompt(scenario['scenario'])
                try:
                    _, activations = self.generate_with_activations(prompt)
                    control_data.append({'activations': activations})
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            all_rollouts_data.append({
                'rollout_idx': rollout_idx,
                'target_data': target_data,
                'control_data': control_data
            })
        
        return {
            'all_rollouts_data': all_rollouts_data,
            'target_foundation': self.target_foundation,
            'control_foundation': self.control_foundation,
            'n_rollouts': self.n_rollouts,
            'model_config': {
                'model_type': self.model_config.model_type,
                'total_layers': self.total_layers,
                'hidden_size': self.model_config.hidden_size
            }
        }

    def extract_differential_vectors_pca(self, experiment_results: Dict) -> Dict:
        """
        Extract differential vectors using pairwise difference + PCA
        
        For each layer:
        1. Each rollout: 200 pairs (target[i] - control[i]) → 200×4096 matrix
        2. Concatenate rollouts: 5 × 200 = 1000×4096 matrix
        3. Extract PC1 and PC2
        """
        print(f"\n{'='*70}")
        print(f"Extracting differential vectors with PCA")
        print(f"{'='*70}\n")
        
        all_rollouts_data = experiment_results['all_rollouts_data']
        first_rollout = all_rollouts_data[0]
        layer_names = list(first_rollout['target_data'][0]['activations'].keys())
        
        differential_vectors = {}
        pca_statistics = {}
        
        for layer_name in layer_names:
            print(f"\nProcessing {layer_name}")
            
            all_diff_vectors = []
            
            for rollout_data in all_rollouts_data:
                target_data = rollout_data['target_data']
                control_data = rollout_data['control_data']
                
                n_pairs = min(len(target_data), len(control_data))
                
                # Create pairwise differences
                for i in range(n_pairs):
                    target_act = target_data[i]['activations'][layer_name].flatten()
                    control_act = control_data[i]['activations'][layer_name].flatten()
                    diff_vector = target_act - control_act
                    all_diff_vectors.append(diff_vector)
            
            # Convert to matrix
            diff_matrix = np.array(all_diff_vectors)  # e.g., 1000 × 4096
            
            print(f"  Differential matrix: {diff_matrix.shape}")
            
            # PCA
            pca = PCA(n_components=2)
            pca.fit(diff_matrix)
            
            pc1 = pca.components_[0]
            pc2 = pca.components_[1]
            explained_variance = pca.explained_variance_ratio_
            
            print(f"  PC1 variance: {explained_variance[0]:.4f}")
            print(f"  PC2 variance: {explained_variance[1]:.4f}")
            
            differential_vectors[layer_name] = {
                'pc1': pc1,
                'pc2': pc2,
                'explained_variance_ratio': explained_variance,
            }
            
            pca_statistics[layer_name] = {
                'n_samples': diff_matrix.shape[0],
                'vector_dim': diff_matrix.shape[1],
                'pc1_variance': float(explained_variance[0]),
                'pc2_variance': float(explained_variance[1]),
                'total_variance': float(explained_variance.sum()),
            }
        
        print(f"\n✅ Extracted PCs for {len(differential_vectors)} layers")
        
        return {
            'differential_vectors': differential_vectors,
            'pca_statistics': pca_statistics,
            'target_foundation': self.target_foundation,
            'control_foundation': self.control_foundation,
            'n_rollouts': experiment_results['n_rollouts'],
            'model_info': experiment_results['model_config']
        }
    
    def save_differential_vectors(self, differential_results: Dict):
        """Save vectors"""
        save_dir = f"{self.output_dir}/differential_vectors"
        base_name = f"{self.model_name}_{self.target_foundation}_vs_{self.control_foundation}"
        
        # Save complete results
        with open(f"{save_dir}/{base_name}_complete.pkl", 'wb') as f:
            pickle.dump(differential_results, f)
        print(f"✅ Saved: {save_dir}/{base_name}_complete.pkl")
        
        # Save PCs
        vectors_dir = f"{save_dir}/vectors_npy"
        os.makedirs(vectors_dir, exist_ok=True)
        
        for layer_name, vector_data in differential_results['differential_vectors'].items():
            safe_layer_name = layer_name.replace('.', '_')
            np.save(f"{vectors_dir}/{base_name}_{safe_layer_name}_pc1.npy", vector_data['pc1'])
            np.save(f"{vectors_dir}/{base_name}_{safe_layer_name}_pc2.npy", vector_data['pc2'])
        
        print(f"✅ Saved PCs: {vectors_dir}/")
        
        # Save statistics
        stats_data = {
            'target_foundation': differential_results['target_foundation'],
            'control_foundation': differential_results['control_foundation'],
            'model_info': differential_results['model_info'],
            'n_rollouts': differential_results['n_rollouts'],
            'pca_statistics': differential_results['pca_statistics']
        }
        
        with open(f"{save_dir}/{base_name}_statistics.json", 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"✅ Saved statistics: {save_dir}/{base_name}_statistics.json")
    
    def cleanup(self):
        """Cleanup"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main():
    parser = argparse.ArgumentParser(description='Differential Vector PCA Experiment')
    
    parser.add_argument('--model_name', type=str, default='mistral-7b-instruct')
    parser.add_argument('--model_path', type=str, 
                      default='/data/cyu/SecAlign/mistral-7b-instruct-v0.1')
    parser.add_argument('--target_foundation', type=str, default='fairness')
    parser.add_argument('--control_foundation', type=str, default='social_norms')
    parser.add_argument('--monitoring_mode', type=str, default='comprehensive')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--data_file', type=str, default='MFV130Gen.json')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=10)
    parser.add_argument('--n_rollouts', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    
    experiment = DifferentialVectorExperiment(
        target_foundation=args.target_foundation,
        control_foundation=args.control_foundation,
        model_name=args.model_name,
        model_path=args.model_path,
        monitoring_mode=args.monitoring_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        n_rollouts=args.n_rollouts,
        random_seed=args.random_seed
    )
    
    try:
        experiment.load_model()
        scenarios_data = experiment.load_moral_scenarios(args.data_file)
        experiment_results = experiment.run_experiment(scenarios_data, args.n_samples)
        differential_results = experiment.extract_differential_vectors_pca(experiment_results)
        experiment.save_differential_vectors(differential_results)
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        experiment.cleanup()


if __name__ == "__main__":
    main()