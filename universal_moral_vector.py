"""
Moral Foundations Concept Vector Experiment - Persona Vector Method
Uses simple mean difference method to extract concept vectors
Supports multiple rollouts per scenario for robust activation collection

Usage:
python concept_vector_experiment.py --model_name mistral-7b-instruct \
    --model_path /path/to/model --target_foundation fairness \
    --control_foundation care --n_rollouts 10
"""

import json
import torch
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import argparse
from datetime import datetime
import csv

warnings.filterwarnings('ignore')


class ConceptVectorExperiment:
    """Concept Vector Experiment Class - Using Persona Vector's simple mean difference method"""
    
    def __init__(self, 
                 target_foundation: str = "fairness",
                 control_foundation: str = "social_norms",
                 model_name: str = "mistral-7b-instruct", 
                 model_path: str = "/data/cyu/SecAlign/mistral-7b-instruct-v0.1", 
                 device: str = "cuda",
                 monitoring_mode: str = "comprehensive",
                 temperature: float = 0.7,
                 max_new_tokens: int = 10,
                 enhanced_monitoring: bool = False,
                 n_rollouts: int = 10):  # ⭐ New parameter
        
        self.target_foundation = target_foundation.lower()
        self.control_foundation = control_foundation.lower()
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.monitoring_mode = monitoring_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.enhanced_monitoring = enhanced_monitoring
        self.n_rollouts = n_rollouts  # ⭐ New attribute
        self.model = None
        self.tokenizer = None
        self.activation_storage = {}
        self.hooks = []
        
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
        
        self.detailed_records = []
        
        print(f"Target foundation: {self.target_foundation.title()}")
        print(f"Control foundation: {self.control_foundation.title()}")
        print(f"Model: {self.model_name}")
        print(f"Method: Simple mean difference (Persona Vector style)")
        print(f"Enhanced monitoring: {'Enabled' if enhanced_monitoring else 'Disabled'}")
        print(f"Temperature: {self.temperature}")
        print(f"Max new tokens: {self.max_new_tokens}")
        print(f"Rollouts per scenario: {self.n_rollouts}")  # ⭐ New print
        print(f"Output directory: {self.output_dir}")
        
    def _setup_output_directory(self) -> str:
        """Setup output directory"""
        if self.target_foundation == self.control_foundation:
            base_name = f"{self.model_name}_{self.target_foundation}_self_control"
        else:
            base_name = f"{self.model_name}_{self.target_foundation}_vs_{self.control_foundation}"
        
        if self.enhanced_monitoring:
            base_name += "_enhanced"
        
        base_name += "_concept_vector"
            
        output_dir = f"MFV130/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(f"{output_dir}/detailed_logs", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{output_dir}/concept_vectors", exist_ok=True)
        
        return output_dir
        
    def _check_model_compatibility(self):
        """Check model compatibility"""
        try:
            print(f"Checking model compatibility: {self.model_path}")
            self.model_config = AutoConfig.from_pretrained(self.model_path)
            self.total_layers = self.model_config.num_hidden_layers
            
            print(f"  Model type: {self.model_config.model_type}")
            print(f"  Hidden layers: {self.total_layers}")
            print(f"  Hidden dimension: {self.model_config.hidden_size}")
            return True
        except Exception as e:
            print(f"❌ Model compatibility check failed: {e}")
            return False
            
    def _configure_monitoring_layers(self) -> List[int]:
        """Configure monitoring layers"""
        if self.total_layers is None:
            total_layers = 32
        else:
            total_layers = self.total_layers
        
        mode_configs = {
            "light": [total_layers//4, total_layers//2, total_layers*3//4],
            "comprehensive": [max(1, total_layers//6), total_layers//3, total_layers//2, 
                             total_layers*2//3, total_layers*5//6, total_layers-2],
            "dense": list(range(2, total_layers, max(1, total_layers//10))),
            "full": list(range(total_layers))
        }
        
        if self.monitoring_mode not in mode_configs:
            print(f"Unknown mode {self.monitoring_mode}, using 'comprehensive'")
            self.monitoring_mode = "comprehensive"
            
        layers = mode_configs[self.monitoring_mode]
        layers = [l for l in layers if 0 <= l < total_layers]
        
        print(f"Total layers: {total_layers}")
        print(f"Target layers: {layers}")
        
        return layers
        
    def load_model(self):
        """Load model"""
        print(f"Loading {self.model_name} from {self.model_path}...")
        
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
                print(f"Loaded {len(unrelated_data['scenarios'])} unrelated questions")
        
        return data
    
    def get_available_foundations(self, scenarios: List[Dict]) -> Dict[str, int]:
        """Get available moral foundations"""
        foundation_counts = {}
        for scenario in scenarios:
            foundation = scenario['foundation']
            foundation_counts[foundation] = foundation_counts.get(foundation, 0) + 1
        return foundation_counts
    
    def filter_scenarios_by_foundation(self, scenarios: List[Dict], foundation_key: str) -> List[Dict]:
        """Filter scenarios"""
        if foundation_key.lower() == "social_norms":
            return [s for s in scenarios if s['foundation'] == 'Social Norms']
        elif foundation_key.lower() == "unrelated":
            return [s for s in scenarios if s['foundation'] == 'Unrelated']
        
        if foundation_key.lower() in self.foundation_mapping:
            target_foundations = self.foundation_mapping[foundation_key.lower()]
            return [s for s in scenarios 
                   if any(s['foundation'].startswith(tf.split()[0]) if '(' in tf 
                         else s['foundation'] == tf 
                         for tf in target_foundations)]
        else:
            return [s for s in scenarios if s['foundation'] == foundation_key]
    
    def create_moral_prompt(self, scenario: str) -> str:
        """Create prompt for moral judgment"""
        prompt = f"""Rate the moral wrongness of this behavior on a 0–4 scale (0 = not at all wrong; 4 = extremely wrong):

"{scenario}"

Rating:"""
        return prompt
    
    def setup_activation_hooks(self, focus_components: List[str] = None):
        """Setup activation collection hooks"""
        if focus_components is None:
            if self.enhanced_monitoring:
                focus_components = [
                    "residual_stream",
                    "self_attn",
                    "self_attn.o_proj",
                    "mlp",
                    "mlp.down_proj",
                ]
            else:
                focus_components = [
                    "self_attn",
                    "self_attn.o_proj",
                    "mlp",
                    "mlp.down_proj",
                ]
        
        self.activation_storage.clear()
        
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        target_module_patterns = []
        for layer_num in self.target_layers:
            for component in focus_components:
                if component == "residual_stream":
                    pattern = f"model.layers.{layer_num}"
                    target_module_patterns.append((pattern, component, layer_num))
                else:
                    pattern = f"model.layers.{layer_num}.{component}"
                    target_module_patterns.append((pattern, component, layer_num))
        
        print(f"Setting up hooks - monitoring {len(focus_components)} components per layer")
        hook_count = 0
        
        for name, module in self.model.named_modules():
            for pattern, component_type, layer_num in target_module_patterns:
                if name == pattern:
                    def make_hook(layer_name, comp_type):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                activation = output[0]
                            else:
                                activation = output
                            
                            if activation.dim() == 3:
                                activation = activation[:, -1, :]
                            
                            if comp_type == "residual_stream":
                                storage_key = f"{layer_name}_residual_stream"
                            else:
                                storage_key = layer_name
                            
                            self.activation_storage[storage_key] = activation.detach().cpu().float().numpy()
                        
                        return hook_fn
                    
                    hook = module.register_forward_hook(make_hook(name, component_type))
                    self.hooks.append(hook)
                    hook_count += 1
                    break
        
        print(f"Successfully registered {hook_count} hooks")
    
    def generate_with_activations(self, prompt: str) -> Tuple[str, Dict]:
        """Generate text and collect activations"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.activation_storage.clear()
        
        # Encoding phase
        with torch.no_grad():
            encoding_outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False
            )
        
        encoding_activations = {k: v.copy() for k, v in self.activation_storage.items()}
        
        # Generation phase
        for hook in self.hooks:
            hook.remove()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        self.setup_activation_hooks()
        
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text, encoding_activations
    
    def generate_with_multiple_rollouts(self, prompt: str, n_rollouts: int) -> Tuple[List[str], List[Dict]]:
        """
        ⭐ NEW: Generate multiple rollouts for a single prompt and collect all activations
        
        Args:
            prompt: The input prompt
            n_rollouts: Number of rollouts to perform
            
        Returns:
            List of generated texts and list of activation dictionaries
        """
        all_generated_texts = []
        all_activations = []
        
        for rollout_idx in range(n_rollouts):
            generated_text, activations = self.generate_with_activations(prompt)
            all_generated_texts.append(generated_text)
            all_activations.append(activations)
        
        return all_generated_texts, all_activations
    
    def _record_detailed_interaction(self, prompt: str, responses: List[str], scenario_data: Dict, 
                                   activations_list: List[Dict], scenario_index: int, 
                                   foundation_type: str, rollout_idx: int = None):
        """
        ⭐ MODIFIED: Record detailed interaction (supports multiple rollouts)
        
        Args:
            rollout_idx: If provided, record single rollout. If None, record all rollouts.
        """
        if rollout_idx is not None:
            # Record single rollout
            record = {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'scenario_index': scenario_index,
                'rollout_index': rollout_idx,
                'foundation_type': foundation_type,
                'original_foundation': scenario_data['foundation'],
                'scenario_text': scenario_data['scenario'],
                'original_wrongness_rating': scenario_data['wrongness_rating'],
                'prompt': prompt,
                'model_response': responses[rollout_idx] if rollout_idx < len(responses) else '',
                'activation_layers': list(activations_list[rollout_idx].keys()) if rollout_idx < len(activations_list) else [],
            }
            
            self.detailed_records.append(record)
        else:
            # Record all rollouts
            for idx in range(len(responses)):
                self._record_detailed_interaction(
                    prompt, responses, scenario_data, activations_list, 
                    scenario_index, foundation_type, rollout_idx=idx
                )
        
        # Save to CSV (save all rollouts)
        csv_file = f"{self.output_dir}/detailed_logs/interactions.csv"
        
        for idx, (response, activations) in enumerate(zip(responses, activations_list)):
            csv_row = {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'scenario_index': scenario_index,
                'rollout_index': idx,
                'foundation_type': foundation_type,
                'original_foundation': scenario_data['foundation'],
                'scenario_text': scenario_data['scenario'],
                'model_response': response,
            }
            
            file_exists = os.path.exists(csv_file)
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)
    
    def run_experiment(self, scenarios_data: Dict, n_samples_per_group: int = 50) -> Dict:
        """
        ⭐ MODIFIED: Run experiment with multiple rollouts per scenario
        """
        print(f"\nStarting experiment: {self.target_foundation.title()} vs {self.control_foundation.title()}")
        print(f"Method: Concept Vector (mean difference)")
        print(f"Rollouts per scenario: {self.n_rollouts}")
        
        available_foundations = self.get_available_foundations(scenarios_data['scenarios'])
        print(f"\nAvailable foundations:")
        for foundation, count in sorted(available_foundations.items()):
            print(f"  - {foundation}: {count} scenarios")
        
        # Filter scenarios
        if self.target_foundation == self.control_foundation:
            all_scenarios = self.filter_scenarios_by_foundation(
                scenarios_data['scenarios'], self.target_foundation
            )
            random.shuffle(all_scenarios)
            total_needed = n_samples_per_group * 2
            if len(all_scenarios) < total_needed:
                n_samples_per_group = len(all_scenarios) // 2
            
            target_scenarios = all_scenarios[:n_samples_per_group]
            control_scenarios = all_scenarios[n_samples_per_group:n_samples_per_group*2]
        else:
            target_scenarios = self.filter_scenarios_by_foundation(
                scenarios_data['scenarios'], self.target_foundation
            )[:n_samples_per_group]
            
            control_scenarios = self.filter_scenarios_by_foundation(
                scenarios_data['scenarios'], self.control_foundation
            )[:n_samples_per_group]
        
        print(f"\n{self.target_foundation.title()} scenarios: {len(target_scenarios)}")
        print(f"{self.control_foundation.title()} scenarios: {len(control_scenarios)}")
        print(f"Total activations to collect: {(len(target_scenarios) + len(control_scenarios)) * self.n_rollouts}")
        
        if len(target_scenarios) == 0 or len(control_scenarios) == 0:
            print("❌ No scenarios found!")
            return {}
        
        self.setup_activation_hooks()
        
        # Collect activations
        target_data = []
        control_data = []
        
        # ⭐ Process target scenarios with multiple rollouts
        print(f"\nProcessing {self.target_foundation.title()} scenarios ({self.n_rollouts} rollouts each)...")
        for i, scenario in enumerate(tqdm(target_scenarios, desc=self.target_foundation.title())):
            prompt = self.create_moral_prompt(scenario['scenario'])
            try:
                # Generate multiple rollouts
                generated_texts, activations_list = self.generate_with_multiple_rollouts(
                    prompt, self.n_rollouts
                )
                
                # Record all rollouts
                self._record_detailed_interaction(
                    prompt, generated_texts, scenario, activations_list, i, self.target_foundation
                )
                
                # Store data with all rollouts
                target_data.append({
                    'scenario': scenario['scenario'],
                    'foundation': scenario['foundation'],
                    'wrongness_rating': scenario['wrongness_rating'],
                    'generated_texts': generated_texts,  # ⭐ Multiple responses
                    'activations_list': activations_list,  # ⭐ Multiple activation sets
                    'n_rollouts': self.n_rollouts
                })
            except Exception as e:
                print(f"Error processing scenario {i}: {e}")
                continue
        
        # ⭐ Process control scenarios with multiple rollouts
        print(f"\nProcessing {self.control_foundation.title()} scenarios ({self.n_rollouts} rollouts each)...")
        for i, scenario in enumerate(tqdm(control_scenarios, desc=self.control_foundation.title())):
            prompt = self.create_moral_prompt(scenario['scenario'])
            try:
                # Generate multiple rollouts
                generated_texts, activations_list = self.generate_with_multiple_rollouts(
                    prompt, self.n_rollouts
                )
                
                # Record all rollouts
                self._record_detailed_interaction(
                    prompt, generated_texts, scenario, activations_list, i, self.control_foundation
                )
                
                # Store data with all rollouts
                control_data.append({
                    'scenario': scenario['scenario'],
                    'foundation': scenario['foundation'],
                    'wrongness_rating': scenario['wrongness_rating'],
                    'generated_texts': generated_texts,  # ⭐ Multiple responses
                    'activations_list': activations_list,  # ⭐ Multiple activation sets
                    'n_rollouts': self.n_rollouts
                })
            except Exception as e:
                print(f"Error processing scenario {i}: {e}")
                continue
        
        print(f"\nSuccessfully processed:")
        print(f"  {self.target_foundation.title()}: {len(target_data)} scenarios × {self.n_rollouts} rollouts = {len(target_data) * self.n_rollouts} total activations")
        print(f"  {self.control_foundation.title()}: {len(control_data)} scenarios × {self.n_rollouts} rollouts = {len(control_data) * self.n_rollouts} total activations")
        
        return {
            'target_data': target_data,
            'control_data': control_data,
            'target_foundation': self.target_foundation,
            'control_foundation': self.control_foundation,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'enhanced_monitoring': self.enhanced_monitoring,
            'n_rollouts': self.n_rollouts,  # ⭐ Store rollouts info
            'model_config': {
                'model_type': self.model_config.model_type if self.model_config else 'unknown',
                'total_layers': self.total_layers,
                'hidden_size': self.model_config.hidden_size if self.model_config else 'unknown'
            },
            'monitoring_config': {
                'mode': self.monitoring_mode,
                'target_layers': self.target_layers,
                'n_target_scenarios': len(target_data),
                'n_control_scenarios': len(control_data),
                'enhanced_monitoring': self.enhanced_monitoring,
                'n_rollouts': self.n_rollouts  # ⭐ Store rollouts info
            }
        }

    def extract_concept_vectors(self, experiment_results: Dict) -> Dict:
        """
        ⭐ MODIFIED: Extract concept vectors from multiple rollouts
        Average activations across all rollouts for each scenario before computing mean difference
        """
        print(f"\n{'='*70}")
        print(f"Extracting concept vectors: {self.target_foundation.title()} vs {self.control_foundation.title()}")
        print(f"Method: Mean difference (Persona Vector)")
        print(f"Rollouts per scenario: {experiment_results.get('n_rollouts', 1)}")
        print(f"{'='*70}\n")
        
        target_data = experiment_results['target_data']
        control_data = experiment_results['control_data']
        
        if not target_data or not control_data:
            print("No data to analyze!")
            return {}
        
        # Get layer names from first rollout of first scenario
        layer_names = list(target_data[0]['activations_list'][0].keys())
        concept_vectors = {}
        vector_statistics = {}
        
        for layer_name in layer_names:
            print(f"\nProcessing layer: {layer_name}")
            
            # ⭐ Collect and average activations across rollouts for each scenario
            target_activations_per_scenario = []
            for data in target_data:
                # Average activations across all rollouts for this scenario
                rollout_activations = []
                for rollout_activations_dict in data['activations_list']:
                    if layer_name in rollout_activations_dict:
                        rollout_activations.append(rollout_activations_dict[layer_name].flatten())
                
                if rollout_activations:
                    # Average across rollouts
                    avg_activation = np.mean(rollout_activations, axis=0)
                    target_activations_per_scenario.append(avg_activation)
            
            control_activations_per_scenario = []
            for data in control_data:
                # Average activations across all rollouts for this scenario
                rollout_activations = []
                for rollout_activations_dict in data['activations_list']:
                    if layer_name in rollout_activations_dict:
                        rollout_activations.append(rollout_activations_dict[layer_name].flatten())
                
                if rollout_activations:
                    # Average across rollouts
                    avg_activation = np.mean(rollout_activations, axis=0)
                    control_activations_per_scenario.append(avg_activation)
            
            target_activations = np.array(target_activations_per_scenario)
            control_activations = np.array(control_activations_per_scenario)
            
            if target_activations.size == 0 or control_activations.size == 0:
                print(f"  ⚠️  Skipping: empty activation data")
                continue
            
            # Core step: calculate mean difference
            print(f"  Target scenarios: {target_activations.shape[0]} (each averaged over {experiment_results.get('n_rollouts', 1)} rollouts)")
            print(f"  Control scenarios: {control_activations.shape[0]} (each averaged over {experiment_results.get('n_rollouts', 1)} rollouts)")
            print(f"  Vector dimension: {target_activations.shape[1]}")
            
            # 1. Calculate target group centroid
            mean_target = np.mean(target_activations, axis=0)
            
            # 2. Calculate control group centroid
            mean_control = np.mean(control_activations, axis=0)
            
            # 3. Calculate difference vector
            concept_vector = mean_target - mean_control
            
            # Calculate vector statistics
            vector_norm = np.linalg.norm(concept_vector)
            vector_norm_l1 = np.linalg.norm(concept_vector, ord=1)
            
            # Calculate normalized vector
            if vector_norm > 1e-8:
                normalized_vector = concept_vector / vector_norm
            else:
                normalized_vector = concept_vector
            
            # Find dimensions with largest absolute values
            top_dims = np.argsort(np.abs(concept_vector))[-10:][::-1]
            
            print(f"  ✅ Vector norm (L2): {vector_norm:.4f}")
            print(f"  ✅ Vector norm (L1): {vector_norm_l1:.4f}")
            print(f"  ✅ Top 5 dimensions: {top_dims[:5].tolist()}")
            
            # Save results
            concept_vectors[layer_name] = {
                'vector': concept_vector,
                'normalized_vector': normalized_vector,
                'mean_target': mean_target,
                'mean_control': mean_control,
            }
            
            vector_statistics[layer_name] = {
                'vector_dim': len(concept_vector),
                'vector_norm_l2': float(vector_norm),
                'vector_norm_l1': float(vector_norm_l1),
                'top_dimensions': top_dims[:10].tolist(),
                'top_values': concept_vector[top_dims[:10]].tolist(),
                'n_target_samples': target_activations.shape[0],
                'n_control_samples': control_activations.shape[0],
                'n_rollouts_per_sample': experiment_results.get('n_rollouts', 1),
                'target_mean_norm': float(np.linalg.norm(mean_target)),
                'control_mean_norm': float(np.linalg.norm(mean_control)),
            }
        
        print(f"\n{'='*70}")
        print(f"✅ Successfully extracted {len(concept_vectors)} concept vectors")
        print(f"{'='*70}\n")
        
        return {
            'concept_vectors': concept_vectors,
            'vector_statistics': vector_statistics,
            'target_foundation': self.target_foundation,
            'control_foundation': self.control_foundation,
            'n_rollouts': experiment_results.get('n_rollouts', 1),
            'model_info': {
                'model_name': self.model_name,
                'model_path': self.model_path,
                'model_type': self.model_config.model_type if self.model_config else 'unknown',
                'total_layers': self.total_layers
            }
        }
    
    def save_concept_vectors(self, concept_results: Dict):
        """Save concept vectors"""
        save_dir = f"{self.output_dir}/concept_vectors"
        
        base_name = f"{self.model_name}_{self.target_foundation}_vs_{self.control_foundation}"
        
        # Save complete results (pickle format)
        with open(f"{save_dir}/{base_name}_complete.pkl", 'wb') as f:
            pickle.dump(concept_results, f)
        
        print(f"✅ Saved complete results to: {save_dir}/{base_name}_complete.pkl")
        
        # Save vectors for each layer (numpy format, convenient for later loading)
        vectors_dir = f"{save_dir}/vectors_npy"
        os.makedirs(vectors_dir, exist_ok=True)
        
        for layer_name, vector_data in concept_results['concept_vectors'].items():
            safe_layer_name = layer_name.replace('.', '_')
            
            # Save original vector
            np.save(
                f"{vectors_dir}/{base_name}_{safe_layer_name}_vector.npy",
                vector_data['vector']
            )
            
            # Save normalized vector
            np.save(
                f"{vectors_dir}/{base_name}_{safe_layer_name}_normalized.npy",
                vector_data['normalized_vector']
            )
        
        print(f"✅ Saved layer vectors to: {vectors_dir}/")
        
        # Save vector statistics (JSON format)
        stats_data = {
            'target_foundation': concept_results['target_foundation'],
            'control_foundation': concept_results['control_foundation'],
            'model_info': concept_results['model_info'],
            'n_rollouts': concept_results.get('n_rollouts', 1),
            'vector_statistics': {}
        }
        
        for layer_name, stats in concept_results['vector_statistics'].items():
            stats_data['vector_statistics'][layer_name] = stats
        
        with open(f"{save_dir}/{base_name}_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved statistics to: {save_dir}/{base_name}_statistics.json")
    
    def visualize_concept_vectors(self, concept_results: Dict):
        """Visualize concept vectors"""
        print(f"\nCreating concept vector visualizations...")
        
        vector_statistics = concept_results['vector_statistics']
        n_rollouts = concept_results.get('n_rollouts', 1)
        
        # Prepare data
        layer_names = []
        vector_norms = []
        layer_numbers = []
        
        for layer_name, stats in vector_statistics.items():
            layer_names.append(layer_name)
            vector_norms.append(stats['vector_norm_l2'])
            
            # Extract layer number
            import re
            match = re.search(r'layers\.(\d+)', layer_name)
            if match:
                layer_numbers.append(int(match.group(1)))
            else:
                layer_numbers.append(0)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        fig.suptitle(
            f'Concept Vectors: {self.target_foundation.title()} vs {self.control_foundation.title()}\n'
            f'Model: {self.model_name} | Rollouts per scenario: {n_rollouts}',
            fontsize=16, fontweight='bold'
        )
        
        # Plot 1: Vector norm trend
        ax1 = axes[0, 0]
        ax1.bar(range(len(layer_names)), vector_norms, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Layers & Components')
        ax1.set_ylabel('Vector Norm (L2)')
        ax1.set_title('Concept Vector Strength Across Layers')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels([ln.replace('model.layers.', 'L') for ln in layer_names], 
                           rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Annotate strongest layer
        if vector_norms:
            max_idx = np.argmax(vector_norms)
            ax1.annotate(f'{vector_norms[max_idx]:.2f}', 
                        xy=(max_idx, vector_norms[max_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        fontsize=10, fontweight='bold')
        
        # Plot 2: Layer depth trend
        ax2 = axes[0, 1]
        if layer_numbers:
            layer_to_norms = {}
            for ln, norm in zip(layer_numbers, vector_norms):
                if ln not in layer_to_norms:
                    layer_to_norms[ln] = []
                layer_to_norms[ln].append(norm)
            
            avg_layer_nums = sorted(layer_to_norms.keys())
            avg_norms = [np.mean(layer_to_norms[ln]) for ln in avg_layer_nums]
            
            ax2.plot(avg_layer_nums, avg_norms, 'o-', linewidth=2, markersize=8,
                    color='green', markerfacecolor='lightgreen')
            ax2.set_xlabel('Layer Number')
            ax2.set_ylabel('Average Vector Norm')
            ax2.set_title('Concept Vector Strength by Layer Depth')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Top dimension distribution
        ax3 = axes[1, 0]
        all_top_dims = []
        for stats in vector_statistics.values():
            all_top_dims.extend(stats['top_dimensions'][:5])
        
        from collections import Counter
        dim_counts = Counter(all_top_dims)
        top_dims = dim_counts.most_common(20)
        
        if top_dims:
            dims, counts = zip(*top_dims)
            ax3.barh(range(len(dims)), counts, alpha=0.7, color='coral')
            ax3.set_yticks(range(len(dims)))
            ax3.set_yticklabels(dims)
            ax3.set_xlabel('Frequency Across Layers')
            ax3.set_ylabel('Neuron Dimension')
            ax3.set_title('Most Important Dimensions (Top 20)')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Vector similarity matrix
        ax4 = axes[1, 1]
        n_layers = len(concept_results['concept_vectors'])
        if n_layers > 1:
            similarity_matrix = np.zeros((n_layers, n_layers))
            vector_list = []
            layer_list = []
            
            for layer_name, vector_data in concept_results['concept_vectors'].items():
                vector_list.append(vector_data['normalized_vector'])
                layer_list.append(layer_name)
            
            for i in range(n_layers):
                for j in range(n_layers):
                    similarity_matrix[i, j] = np.dot(vector_list[i], vector_list[j])
            
            im = ax4.imshow(similarity_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
            ax4.set_xticks(range(n_layers))
            ax4.set_yticks(range(n_layers))
            ax4.set_xticklabels([ln.replace('model.layers.', 'L') for ln in layer_list], 
                               rotation=45, ha='right', fontsize=8)
            ax4.set_yticklabels([ln.replace('model.layers.', 'L') for ln in layer_list], 
                               fontsize=8)
            ax4.set_title('Cosine Similarity Between Layer Vectors')
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        save_path = f"{self.output_dir}/visualizations/{self.model_name}_{self.target_foundation}_vs_{self.control_foundation}_concept_vectors.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to: {save_path}")
    
    def generate_report(self, concept_results: Dict) -> str:
        """Generate analysis report"""
        vector_statistics = concept_results['vector_statistics']
        model_info = concept_results['model_info']
        n_rollouts = concept_results.get('n_rollouts', 1)
        
        report = []
        report.append("=" * 80)
        report.append(f"Concept Vector Extraction Report - Persona Vector Method")
        report.append(f"{self.target_foundation.title()} vs {self.control_foundation.title()}")
        report.append("=" * 80)
        
        # Basic information
        report.append(f"\nModel Information:")
        report.append(f"  - Model name: {model_info['model_name']}")
        report.append(f"  - Model type: {model_info['model_type']}")
        report.append(f"  - Total layers: {model_info['total_layers']}")
        report.append(f"  - Target foundation: {self.target_foundation.title()}")
        report.append(f"  - Control foundation: {self.control_foundation.title()}")
        report.append(f"  - Temperature: {self.temperature}")
        report.append(f"  - Max new tokens: {self.max_new_tokens}")
        report.append(f"  - Rollouts per scenario: {n_rollouts}")
        report.append(f"  - Output directory: {self.output_dir}")
        
        # Method explanation
        report.append(f"\nMethod:")
        report.append(f"  - Extraction method: Mean difference (mean_target - mean_control)")
        report.append(f"  - Rollout strategy: {n_rollouts} rollouts per scenario, averaged before mean difference")
        report.append(f"  - Statistical testing: None (simplified version)")
        report.append(f"  - Number of vectors: {len(vector_statistics)}")
        
        # Vector statistics
        report.append(f"\nVector Statistics:")
        
        # Find strongest vector
        max_norm_layer = max(vector_statistics.items(), 
                            key=lambda x: x[1]['vector_norm_l2'])
        
        report.append(f"  - Strongest vector layer: {max_norm_layer[0]}")
        report.append(f"  - Strongest vector norm: {max_norm_layer[1]['vector_norm_l2']:.4f}")
        report.append(f"  - Average vector norm: {np.mean([s['vector_norm_l2'] for s in vector_statistics.values()]):.4f}")
        
        # Top 5 layers
        report.append(f"\nTop 5 Strongest Concept Vectors:")
        sorted_layers = sorted(vector_statistics.items(), 
                              key=lambda x: x[1]['vector_norm_l2'], 
                              reverse=True)
        
        for i, (layer_name, stats) in enumerate(sorted_layers[:5]):
            report.append(f"  {i+1}. {layer_name}")
            report.append(f"     - L2 norm: {stats['vector_norm_l2']:.4f}")
            report.append(f"     - Target scenarios: {stats['n_target_samples']} (× {n_rollouts} rollouts)")
            report.append(f"     - Control scenarios: {stats['n_control_samples']} (× {n_rollouts} rollouts)")
        
        # Output files
        report.append(f"\nOutput Files:")
        report.append(f"  - Vector data: {self.output_dir}/concept_vectors/")
        report.append(f"  - Visualizations: {self.output_dir}/visualizations/")
        report.append(f"  - Detailed logs: {self.output_dir}/detailed_logs/")
        
        # Usage recommendations
        report.append(f"\nUsage Recommendations:")
        report.append(f"  - Load vector: np.load('vectors_npy/xxx_vector.npy')")
        report.append(f"  - Steering: h = h + alpha * vector")
        report.append(f"  - Monitoring: projection = activation @ vector")
        report.append(f"  - Recommended alpha range: 0.5 - 2.0")
        
        return "\n".join(report)
    
    def cleanup(self):
        """Cleanup resources"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Concept Vector Experiment')
    
    parser.add_argument('--model_name', type=str, default='mistral-7b-instruct')
    parser.add_argument('--model_path', type=str, 
                      default='/data/cyu/SecAlign/mistral-7b-instruct-v0.1')
    parser.add_argument('--target_foundation', type=str, default='fairness',
                      choices=['care', 'fairness', 'loyalty', 'authority', 
                              'sanctity', 'liberty', 'unrelated'])
    parser.add_argument('--control_foundation', type=str, default='social_norms',
                      choices=['care', 'fairness', 'loyalty', 'authority', 
                              'sanctity', 'liberty', 'unrelated', 'social_norms'])
    parser.add_argument('--monitoring_mode', type=str, default='comprehensive',
                      choices=['light', 'comprehensive', 'dense', 'full'])
    parser.add_argument('--enhanced_monitoring', action='store_true')
    parser.add_argument('--n_samples', type=int, default=30)
    parser.add_argument('--data_file', type=str, default='MFV130Gen.json')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation (default: 0.7)')
    parser.add_argument('--max_new_tokens', type=int, default=10,
                        help='Maximum number of new tokens to generate (default: 10)')
    parser.add_argument('--n_rollouts', type=int, default=10,  
                        help='Number of rollouts per scenario (default: 10)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"Concept Vector Extraction Experiment")
    print(f"{args.target_foundation.title()} vs {args.control_foundation.title()}")
    print(f"Method: Persona Vector (mean difference)")
    print(f"Temperature: {args.temperature}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Samples per group: {args.n_samples}")
    print(f"Rollouts per scenario: {args.n_rollouts}") 
    print("="*70 + "\n")
    
    # Initialize experiment
    experiment = ConceptVectorExperiment(
        target_foundation=args.target_foundation,
        control_foundation=args.control_foundation,
        model_name=args.model_name,
        model_path=args.model_path,
        monitoring_mode=args.monitoring_mode,
        enhanced_monitoring=args.enhanced_monitoring,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        n_rollouts=args.n_rollouts  
    )
    
    try:
        # Load model
        experiment.load_model()
        
        # Load data
        scenarios_data = experiment.load_moral_scenarios(args.data_file)
        
        # Run experiment
        experiment_results = experiment.run_experiment(scenarios_data, 
                                                      n_samples_per_group=args.n_samples)
        
        if not experiment_results:
            print("Experiment failed")
            return
        
        # Extract concept vectors
        concept_results = experiment.extract_concept_vectors(experiment_results)
        
        if concept_results:
            # Save vectors
            experiment.save_concept_vectors(concept_results)
            
            # Visualize
            experiment.visualize_concept_vectors(concept_results)
            
            # Generate report
            report = experiment.generate_report(concept_results)
            print("\n" + report)
            
            # Save report
            base_name = f"{args.model_name}_{args.target_foundation}_vs_{args.control_foundation}"
            report_path = f"{experiment.output_dir}/concept_vectors/{base_name}_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            
            print(f"\n✅ Report saved to: {report_path}")
            print(f"✅ All results saved to: {experiment.output_dir}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        experiment.cleanup()


if __name__ == "__main__":
    main()