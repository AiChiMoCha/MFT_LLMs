"""
道德基础概念向量实验 - Persona Vector方法
使用简单的均值差方法提取概念向量

使用方法：
python concept_vector_experiment.py --model_name mistral-7b-instruct \
    --model_path /path/to/model --target_foundation fairness \
    --control_foundation care
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
    """概念向量实验类 - 使用Persona Vector的简单均值差方法"""
    
    def __init__(self, 
                 target_foundation: str = "fairness",
                 control_foundation: str = "social_norms",
                 model_name: str = "mistral-7b-instruct", 
                 model_path: str = "/data/cyu/SecAlign/mistral-7b-instruct-v0.1", 
                 device: str = "cuda",
                 monitoring_mode: str = "comprehensive",
                 temperature: float = 0.7,
                 max_new_tokens: int = 10,
                 enhanced_monitoring: bool = False):
        
        self.target_foundation = target_foundation.lower()
        self.control_foundation = control_foundation.lower()
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.monitoring_mode = monitoring_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.enhanced_monitoring = enhanced_monitoring
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
        print(f"Output directory: {self.output_dir}")
        
    def _setup_output_directory(self) -> str:
        """设置输出目录"""
        if self.target_foundation == self.control_foundation:
            base_name = f"{self.model_name}_{self.target_foundation}_self_control"
        else:
            base_name = f"{self.model_name}_{self.target_foundation}_vs_{self.control_foundation}"
        
        if self.enhanced_monitoring:
            base_name += "_enhanced"
        
        base_name += "_concept_vector"  # 标记使用concept vector方法
            
        output_dir = f"MFV130/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子文件夹
        os.makedirs(f"{output_dir}/detailed_logs", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{output_dir}/concept_vectors", exist_ok=True)  # 新增：保存向量
        
        return output_dir
        
    def _check_model_compatibility(self):
        """检查模型兼容性"""
        try:
            print(f"检查模型兼容性: {self.model_path}")
            self.model_config = AutoConfig.from_pretrained(self.model_path)
            self.total_layers = self.model_config.num_hidden_layers
            
            print(f"  模型类型: {self.model_config.model_type}")
            print(f"  隐藏层数: {self.total_layers}")
            print(f"  隐藏维度: {self.model_config.hidden_size}")
            return True
        except Exception as e:
            print(f"❌ 模型兼容性检查失败: {e}")
            return False
            
    def _configure_monitoring_layers(self) -> List[int]:
        """配置监控层"""
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
        """加载模型"""
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
        """加载道德情境数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if os.path.exists(unrelated_path):
            with open(unrelated_path, 'r', encoding='utf-8') as f:
                unrelated_data = json.load(f)
                data['scenarios'].extend(unrelated_data['scenarios'])
                print(f"Loaded {len(unrelated_data['scenarios'])} unrelated questions")
        
        return data
    
    def get_available_foundations(self, scenarios: List[Dict]) -> Dict[str, int]:
        """获取可用的道德基础"""
        foundation_counts = {}
        for scenario in scenarios:
            foundation = scenario['foundation']
            foundation_counts[foundation] = foundation_counts.get(foundation, 0) + 1
        return foundation_counts
    
    def filter_scenarios_by_foundation(self, scenarios: List[Dict], foundation_key: str) -> List[Dict]:
        """筛选情境"""
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
        """创建道德判断的prompt"""
        prompt = f"""Rate the moral wrongness of this behavior on a 0–4 scale (0 = not at all wrong; 4 = extremely wrong):

"{scenario}"

Rating:"""
        return prompt
    
    def setup_activation_hooks(self, focus_components: List[str] = None):
        """设置激活收集hooks"""
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
        
        print(f"设置hooks - 每层监控 {len(focus_components)} 个组件")
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
        
        print(f"成功注册 {hook_count} 个hooks")
    
    def generate_with_activations(self, prompt: str) -> Tuple[str, Dict]:
        """生成文本并收集激活"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.activation_storage.clear()
        
        # 编码阶段
        with torch.no_grad():
            encoding_outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False
            )
        
        encoding_activations = {k: v.copy() for k, v in self.activation_storage.items()}
        
        # 生成阶段
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
    
    def _record_detailed_interaction(self, prompt: str, response: str, scenario_data: Dict, 
                                   activations: Dict, scenario_index: int, foundation_type: str):
        """记录详细交互"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'scenario_index': scenario_index,
            'foundation_type': foundation_type,
            'original_foundation': scenario_data['foundation'],
            'scenario_text': scenario_data['scenario'],
            'original_wrongness_rating': scenario_data['wrongness_rating'],
            'prompt': prompt,
            'model_response': response,
            'activation_layers': list(activations.keys()),
        }
        
        self.detailed_records.append(record)
        
        # 保存到CSV
        csv_file = f"{self.output_dir}/detailed_logs/interactions.csv"
        csv_row = {
            'timestamp': record['timestamp'],
            'model_name': record['model_name'],
            'scenario_index': record['scenario_index'],
            'foundation_type': record['foundation_type'],
            'original_foundation': record['original_foundation'],
            'scenario_text': record['scenario_text'],
            'model_response': record['model_response'],
        }
        
        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)
    
    def run_experiment(self, scenarios_data: Dict, n_samples_per_group: int = 50) -> Dict:
        """运行实验"""
        print(f"\nStarting experiment: {self.target_foundation.title()} vs {self.control_foundation.title()}")
        print(f"Method: Concept Vector (mean difference)")
        
        available_foundations = self.get_available_foundations(scenarios_data['scenarios'])
        print(f"\nAvailable foundations:")
        for foundation, count in sorted(available_foundations.items()):
            print(f"  - {foundation}: {count} scenarios")
        
        # 筛选场景
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
        
        if len(target_scenarios) == 0 or len(control_scenarios) == 0:
            print("❌ No scenarios found!")
            return {}
        
        self.setup_activation_hooks()
        
        # 收集激活
        target_data = []
        control_data = []
        
        print(f"\nProcessing {self.target_foundation.title()} scenarios...")
        for i, scenario in enumerate(tqdm(target_scenarios, desc=self.target_foundation.title())):
            prompt = self.create_moral_prompt(scenario['scenario'])
            try:
                generated_text, activations = self.generate_with_activations(prompt)
                
                self._record_detailed_interaction(
                    prompt, generated_text, scenario, activations, i, self.target_foundation
                )
                
                target_data.append({
                    'scenario': scenario['scenario'],
                    'foundation': scenario['foundation'],
                    'wrongness_rating': scenario['wrongness_rating'],
                    'generated_text': generated_text,
                    'activations': activations
                })
            except Exception as e:
                print(f"Error processing scenario {i}: {e}")
                continue
        
        print(f"\nProcessing {self.control_foundation.title()} scenarios...")
        for i, scenario in enumerate(tqdm(control_scenarios, desc=self.control_foundation.title())):
            prompt = self.create_moral_prompt(scenario['scenario'])
            try:
                generated_text, activations = self.generate_with_activations(prompt)
                
                self._record_detailed_interaction(
                    prompt, generated_text, scenario, activations, i, self.control_foundation
                )
                
                control_data.append({
                    'scenario': scenario['scenario'],
                    'foundation': scenario['foundation'],
                    'wrongness_rating': scenario['wrongness_rating'],
                    'generated_text': generated_text,
                    'activations': activations
                })
            except Exception as e:
                print(f"Error processing scenario {i}: {e}")
                continue
        
        print(f"\nSuccessfully processed:")
        print(f"  {self.target_foundation.title()}: {len(target_data)}")
        print(f"  {self.control_foundation.title()}: {len(control_data)}")
        
        return {
            'target_data': target_data,
            'control_data': control_data,
            'target_foundation': self.target_foundation,
            'control_foundation': self.control_foundation,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'enhanced_monitoring': self.enhanced_monitoring,
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
                'enhanced_monitoring': self.enhanced_monitoring
            }
        }

    def extract_concept_vectors(self, experiment_results: Dict) -> Dict:
        """
        提取概念向量 - 使用简单的均值差方法
        这是核心改动：不做统计检验，直接计算两组激活的均值差
        """
        print(f"\n{'='*70}")
        print(f"提取概念向量: {self.target_foundation.title()} vs {self.control_foundation.title()}")
        print(f"方法: 均值差 (Persona Vector)")
        print(f"{'='*70}\n")
        
        target_data = experiment_results['target_data']
        control_data = experiment_results['control_data']
        
        if not target_data or not control_data:
            print("No data to analyze!")
            return {}
        
        layer_names = list(target_data[0]['activations'].keys())
        concept_vectors = {}
        vector_statistics = {}
        
        for layer_name in layer_names:
            print(f"\n处理层: {layer_name}")
            
            # 收集激活矩阵
            target_activations = np.array([
                data['activations'][layer_name].flatten() 
                for data in target_data 
                if layer_name in data['activations']
            ])
            
            control_activations = np.array([
                data['activations'][layer_name].flatten() 
                for data in control_data 
                if layer_name in data['activations']
            ])
            
            if target_activations.size == 0 or control_activations.size == 0:
                print(f"  ⚠️  跳过：激活数据为空")
                continue
            
            # 核心步骤：计算均值差
            print(f"  目标组样本数: {target_activations.shape[0]}")
            print(f"  对照组样本数: {control_activations.shape[0]}")
            print(f"  向量维度: {target_activations.shape[1]}")
            
            # 1. 计算目标组中心点
            mean_target = np.mean(target_activations, axis=0)
            
            # 2. 计算对照组中心点
            mean_control = np.mean(control_activations, axis=0)
            
            # 3. 计算差异向量
            concept_vector = mean_target - mean_control
            
            # 计算向量统计信息
            vector_norm = np.linalg.norm(concept_vector)
            vector_norm_l1 = np.linalg.norm(concept_vector, ord=1)
            
            # 计算归一化向量
            if vector_norm > 1e-8:
                normalized_vector = concept_vector / vector_norm
            else:
                normalized_vector = concept_vector
            
            # 计算向量方向上最大的维度
            top_dims = np.argsort(np.abs(concept_vector))[-10:][::-1]
            
            print(f"  ✅ 向量范数 (L2): {vector_norm:.4f}")
            print(f"  ✅ 向量范数 (L1): {vector_norm_l1:.4f}")
            print(f"  ✅ Top 5 维度: {top_dims[:5].tolist()}")
            
            # 保存结果
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
                'target_mean_norm': float(np.linalg.norm(mean_target)),
                'control_mean_norm': float(np.linalg.norm(mean_control)),
            }
        
        print(f"\n{'='*70}")
        print(f"✅ 成功提取 {len(concept_vectors)} 个概念向量")
        print(f"{'='*70}\n")
        
        return {
            'concept_vectors': concept_vectors,
            'vector_statistics': vector_statistics,
            'target_foundation': self.target_foundation,
            'control_foundation': self.control_foundation,
            'model_info': {
                'model_name': self.model_name,
                'model_path': self.model_path,
                'model_type': self.model_config.model_type if self.model_config else 'unknown',
                'total_layers': self.total_layers
            }
        }
    
    def save_concept_vectors(self, concept_results: Dict):
        """保存概念向量"""
        save_dir = f"{self.output_dir}/concept_vectors"
        
        base_name = f"{self.model_name}_{self.target_foundation}_vs_{self.control_foundation}"
        
        # 保存完整结果（pickle格式）
        with open(f"{save_dir}/{base_name}_complete.pkl", 'wb') as f:
            pickle.dump(concept_results, f)
        
        print(f"✅ 保存完整结果到: {save_dir}/{base_name}_complete.pkl")
        
        # 保存每层的向量（numpy格式，方便后续加载）
        vectors_dir = f"{save_dir}/vectors_npy"
        os.makedirs(vectors_dir, exist_ok=True)
        
        for layer_name, vector_data in concept_results['concept_vectors'].items():
            safe_layer_name = layer_name.replace('.', '_')
            
            # 保存原始向量
            np.save(
                f"{vectors_dir}/{base_name}_{safe_layer_name}_vector.npy",
                vector_data['vector']
            )
            
            # 保存归一化向量
            np.save(
                f"{vectors_dir}/{base_name}_{safe_layer_name}_normalized.npy",
                vector_data['normalized_vector']
            )
        
        print(f"✅ 保存各层向量到: {vectors_dir}/")
        
        # 保存向量统计信息（JSON格式）
        stats_data = {
            'target_foundation': concept_results['target_foundation'],
            'control_foundation': concept_results['control_foundation'],
            'model_info': concept_results['model_info'],
            'vector_statistics': {}
        }
        
        for layer_name, stats in concept_results['vector_statistics'].items():
            stats_data['vector_statistics'][layer_name] = stats
        
        with open(f"{save_dir}/{base_name}_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 保存统计信息到: {save_dir}/{base_name}_statistics.json")
    
    def visualize_concept_vectors(self, concept_results: Dict):
        """可视化概念向量"""
        print(f"\n创建概念向量可视化...")
        
        vector_statistics = concept_results['vector_statistics']
        
        # 准备数据
        layer_names = []
        vector_norms = []
        layer_numbers = []
        
        for layer_name, stats in vector_statistics.items():
            layer_names.append(layer_name)
            vector_norms.append(stats['vector_norm_l2'])
            
            # 提取层号
            import re
            match = re.search(r'layers\.(\d+)', layer_name)
            if match:
                layer_numbers.append(int(match.group(1)))
            else:
                layer_numbers.append(0)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        fig.suptitle(
            f'Concept Vectors: {self.target_foundation.title()} vs {self.control_foundation.title()}\n'
            f'Model: {self.model_name}',
            fontsize=16, fontweight='bold'
        )
        
        # 图1：向量范数趋势
        ax1 = axes[0, 0]
        ax1.bar(range(len(layer_names)), vector_norms, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Layers & Components')
        ax1.set_ylabel('Vector Norm (L2)')
        ax1.set_title('Concept Vector Strength Across Layers')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels([ln.replace('model.layers.', 'L') for ln in layer_names], 
                           rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 标注最强的层
        if vector_norms:
            max_idx = np.argmax(vector_norms)
            ax1.annotate(f'{vector_norms[max_idx]:.2f}', 
                        xy=(max_idx, vector_norms[max_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        fontsize=10, fontweight='bold')
        
        # 图2：层级趋势（如果有层号信息）
        ax2 = axes[0, 1]
        if layer_numbers:
            # 按层号分组平均
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
        
        # 图3：Top维度分布
        ax3 = axes[1, 0]
        
        # 收集所有层的top维度
        all_top_dims = []
        for stats in vector_statistics.values():
            all_top_dims.extend(stats['top_dimensions'][:5])
        
        # 统计频率
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
        
        # 图4：向量相似度矩阵（余弦相似度）
        ax4 = axes[1, 1]
        
        # 计算层间相似度
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
        plt.show()
        
        print(f"✅ 可视化保存到: {save_path}")
    
    def generate_report(self, concept_results: Dict) -> str:
        """生成分析报告"""
        vector_statistics = concept_results['vector_statistics']
        model_info = concept_results['model_info']
        
        report = []
        report.append("=" * 80)
        report.append(f"概念向量提取报告 - Persona Vector方法")
        report.append(f"{self.target_foundation.title()} vs {self.control_foundation.title()}")
        report.append("=" * 80)
        
        # 基本信息
        report.append(f"\n模型信息:")
        report.append(f"  - 模型名称: {model_info['model_name']}")
        report.append(f"  - 模型类型: {model_info['model_type']}")
        report.append(f"  - 总层数: {model_info['total_layers']}")
        report.append(f"  - 目标基础: {self.target_foundation.title()}")
        report.append(f"  - 对照基础: {self.control_foundation.title()}")
        report.append(f"  - Temperature: {self.temperature}")
        report.append(f"  - Max new tokens: {self.max_new_tokens}")
        report.append(f"  - 输出目录: {self.output_dir}")
        
        # 方法说明
        report.append(f"\n方法:")
        report.append(f"  - 提取方式: 均值差 (mean_target - mean_control)")
        report.append(f"  - 统计检验: 无（简化版）")
        report.append(f"  - 向量数量: {len(vector_statistics)}")
        
        # 向量统计
        report.append(f"\n向量统计:")
        
        # 找出最强的向量
        max_norm_layer = max(vector_statistics.items(), 
                            key=lambda x: x[1]['vector_norm_l2'])
        
        report.append(f"  - 最强向量层: {max_norm_layer[0]}")
        report.append(f"  - 最强向量范数: {max_norm_layer[1]['vector_norm_l2']:.4f}")
        report.append(f"  - 平均向量范数: {np.mean([s['vector_norm_l2'] for s in vector_statistics.values()]):.4f}")
        
        # Top 5 层
        report.append(f"\nTop 5 最强概念向量:")
        sorted_layers = sorted(vector_statistics.items(), 
                              key=lambda x: x[1]['vector_norm_l2'], 
                              reverse=True)
        
        for i, (layer_name, stats) in enumerate(sorted_layers[:5]):
            report.append(f"  {i+1}. {layer_name}")
            report.append(f"     - L2范数: {stats['vector_norm_l2']:.4f}")
            report.append(f"     - 目标组样本: {stats['n_target_samples']}")
            report.append(f"     - 对照组样本: {stats['n_control_samples']}")
        
        # 输出文件
        report.append(f"\n输出文件:")
        report.append(f"  - 向量数据: {self.output_dir}/concept_vectors/")
        report.append(f"  - 可视化: {self.output_dir}/visualizations/")
        report.append(f"  - 详细日志: {self.output_dir}/detailed_logs/")
        
        # 使用建议
        report.append(f"\n使用建议:")
        report.append(f"  - 加载向量: np.load('vectors_npy/xxx_vector.npy')")
        report.append(f"  - Steering: h = h + alpha * vector")
        report.append(f"  - 监控: projection = activation @ vector")
        report.append(f"  - 推荐alpha范围: 0.5 - 2.0")
        
        return "\n".join(report)
    
    def cleanup(self):
        """清理资源"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main():
    """主函数"""
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
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"概念向量提取实验")
    print(f"{args.target_foundation.title()} vs {args.control_foundation.title()}")
    print(f"方法: Persona Vector (均值差)")
    print(f"Temperature: {args.temperature}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Samples per group: {args.n_samples}")
    print("="*70 + "\n")
    
    # 初始化实验
    experiment = ConceptVectorExperiment(
        target_foundation=args.target_foundation,
        control_foundation=args.control_foundation,
        model_name=args.model_name,
        model_path=args.model_path,
        monitoring_mode=args.monitoring_mode,
        enhanced_monitoring=args.enhanced_monitoring,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    
    try:
        # 加载模型
        experiment.load_model()
        
        # 加载数据
        scenarios_data = experiment.load_moral_scenarios(args.data_file)
        
        # 运行实验
        experiment_results = experiment.run_experiment(scenarios_data, 
                                                      n_samples_per_group=args.n_samples)
        
        if not experiment_results:
            print("实验失败")
            return
        
        # 提取概念向量
        concept_results = experiment.extract_concept_vectors(experiment_results)
        
        if concept_results:
            # 保存向量
            experiment.save_concept_vectors(concept_results)
            
            # 可视化
            experiment.visualize_concept_vectors(concept_results)
            
            # 生成报告
            report = experiment.generate_report(concept_results)
            print("\n" + report)
            
            # 保存报告
            base_name = f"{args.model_name}_{args.target_foundation}_vs_{args.control_foundation}"
            report_path = f"{experiment.output_dir}/concept_vectors/{base_name}_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            
            print(f"\n✅ 报告保存到: {report_path}")
            print(f"✅ 所有结果保存到: {experiment.output_dir}")
        
    except Exception as e:
        print(f"实验失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        experiment.cleanup()


if __name__ == "__main__":
    main()